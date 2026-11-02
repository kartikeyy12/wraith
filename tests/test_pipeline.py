"""Pipeline tests on synthetic data. Run from the repo root:  python tests/test_pipeline.py"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("logger", "dsp", "tools"):
    sys.path.insert(0, str(ROOT / "host" / sub))

import csi_eval as ev  # noqa: E402
import csi_features as ft  # noqa: E402
from csi_io import load_session, unwrap_device_time  # noqa: E402
from csi_parser import parse_csi_line  # noqa: E402
import synth_csi  # noqa: E402

DAYS = ["2026-01-01", "2026-01-02", "2026-01-03"]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        synth_csi.generate(cls.dir, DAYS, seed=1, duration_s=60, rate_hz=50)
        cls.paths = ev.discover(cls.dir)
        cls.cfg = ft.FeatureConfig()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestParser(unittest.TestCase):
    S3 = 'CSI_DATA,7,1a:00:00:00:00:00,-40,11,1,7,1,1,1,0,0,0,0,-90,0,11,2,123456,0,66,0,4,0,"[1,2,3,4]"'
    C6 = 'CSI_DATA,7,1a:00:00:00:00:00,-40,11,-90,10,20,11,123456,66,0,4,0,"[1,2,3,4]"'

    def test_s3(self):
        r = parse_csi_line(self.S3)
        self.assertEqual((r["seq"], r["rssi"], r["bandwidth"], r["raw"]), (7, -40, 1, [1, 2, 3, 4]))

    def test_c6_layout(self):
        r = parse_csi_line(self.C6)
        self.assertEqual((r["noise_floor"], r["sig_mode"], r["channel"]), (-90, None, 11))

    def test_non_csi_and_malformed(self):
        self.assertIsNone(parse_csi_line("I (123) wifi: hello"))
        self.assertIn("error", parse_csi_line('CSI_DATA,1,2,3,"[1,2]"'))
        self.assertIn("error", parse_csi_line(self.S3.replace("-40", "x")))


class TestLoggerAndLoader(Base):
    def test_sidecar_and_stats(self):
        m = json.loads((self.dir / f"{DAYS[0]}_zoneA_moving_0.meta.json").read_text())
        self.assertEqual((m["label"], m["activity"], m["day"], m["status"]), ("zoneA", "moving", DAYS[0], "complete"))
        self.assertGreater(m["packets_lost_est"], 0)           # the generator drops ~2% of packets
        self.assertAlmostEqual(m["packet_loss_pct_est"], 2.0, delta=1.5)

    def test_loader_shapes_and_clock_wrap(self):
        s = load_session(self.dir / f"{DAYS[0]}_zoneB_still_0.csv")
        self.assertEqual(s.lltf.shape[1], 64)
        self.assertTrue(np.iscomplexobj(s.lltf))
        self.assertTrue((np.diff(s.t) > 0).all())
        self.assertAlmostEqual(s.t[-1], 60.0, delta=1.0)       # device clock wraps 3 s into the session
        self.assertTrue(np.allclose(s.lltf[:, [0, 27, 33, 37]], 0))  # DC and guard bins are empty

    def test_unwrap(self):
        t = unwrap_device_time(np.array([2 ** 32 - 10, 5, 20]))
        self.assertTrue(np.allclose(t, [0, 15e-6, 30e-6]))


class TestFeatures(Base):
    def test_hampel_removes_spike_only(self):
        x = np.sin(np.linspace(0, 6, 200)) * 0.1 + 1
        y = x.copy(); y[100] += 5
        h = ft.hampel(y, 7, 5.0)
        self.assertLess(abs(h[100] - x[100]), 0.1)
        self.assertTrue(np.allclose(np.delete(h, 100), np.delete(x, 100), atol=1e-9))

    def test_gap_flagged(self):
        t = np.concatenate([np.arange(0, 5, 0.02), np.arange(6, 10, 0.02)])
        g, _, ok = ft.resample_uniform(t, np.ones((len(t), 2)), 50, 0.25)
        self.assertFalse(ok[(g > 5.1) & (g < 5.9)].any())
        self.assertTrue(ok[g < 4.9].all())

    def test_window_vector_consistency(self):
        s = load_session(self.dir / f"{DAYS[0]}_zoneA_moving_0.csv")
        X, ts, names = ft.session_windows(s, self.cfg)
        self.assertEqual(X.shape[1], len(names))
        self.assertGreater(len(X), 15)
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(sum(n.startswith("fp_") for n in names), 48)

    def test_nbvi_select(self):
        A = np.abs(np.random.default_rng(0).normal(10, 1, (500, 20)))
        A[:, 3] += np.random.default_rng(1).normal(0, 8, 500)   # one very unstable subcarrier
        sel = ft.nbvi_select(A, 5)
        self.assertEqual(len(sel), 5)
        self.assertNotIn(3, sel)


class TestEvaluation(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.T = ev.build_table(cls.paths, cls.cfg, baseline_ref=False)
        cls.Tb = ev.build_table(cls.paths, cls.cfg, baseline_ref=True)

    def test_baseline_sessions_excluded(self):
        self.assertNotIn("baseline", set(self.T.activity))

    def test_motion_task_is_solved_by_dynamics(self):
        r = ev.evaluate(self.T, "day", "motion", ("rf", "threshold"), "motion")
        for m in ("rf", "threshold"):
            self.assertGreater(r[m]["pooled"]["bal_acc"], 0.95, m)
            self.assertLess(r[m]["fpr_empty"], 0.1, m)

    def test_motion_threshold_is_blind_to_still_person(self):
        # Presence labels include people standing still, which a motion/variance detector cannot see.
        r = ev.evaluate(self.T, "day", "presence", ("threshold",), "motion")["threshold"]
        self.assertGreater(r["recall_by_activity"]["moving"], 0.9)
        self.assertLess(r["recall_by_activity"]["still"], 0.6)

    def test_level_features_are_day_dependent(self):
        # Regression test for the bug found while building this: level features must not be in the default groups.
        names = self.T.names
        for g in ("motion", "both"):
            cols = ev.feature_columns(names, g)
            self.assertFalse(any(names[i] in ev.LEVEL for i in cols), g)
        self.assertTrue(any(names[i] in ev.LEVEL for i in ev.feature_columns(names, "all")))

    def test_leak_guard(self):
        T = self.T
        tr = np.flatnonzero(T.session == T.session[0])
        with self.assertRaises(RuntimeError):
            ev._check_no_leak("session", T, tr, tr)

    def test_day_and_temporal_splits(self):
        r = ev.evaluate(self.T, "day", "zone", ("rf",), "both")
        self.assertEqual(len(r["rf"]["folds"]), 3)
        r = ev.evaluate(self.T, "temporal", "zone", ("rf",), "both")
        self.assertEqual([t["n_train_days"] for t in r["rf"]["temporal"]], [1, 2])

    def test_baseline_reference_removes_drift(self):
        raw = ev.evaluate(self.T, "day", "zone", ("rf",), "fp")["rf"]["pooled"]["bal_acc"]
        ref = ev.evaluate(self.Tb, "day", "zone", ("rf",), "fp")["rf"]["pooled"]["bal_acc"]
        self.assertGreater(ref, raw + 0.1, f"baseline-referenced {ref:.2f} vs raw {raw:.2f}")

    def test_baseline_referenced_fingerprint_sees_still_person(self):
        r = ev.evaluate(self.Tb, "day", "presence", ("rf",), "fp")["rf"]
        self.assertGreater(r["recall_by_activity"]["still"], 0.8)
        self.assertLess(r["fpr_empty"], 0.2)

    def test_report_formats(self):
        txt = ev.format_results(ev.evaluate(self.T, "session", "zone", ("rf",), "both"), "t")
        self.assertIn("balanced acc", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
