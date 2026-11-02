"""
Leakage-safe evaluation of CSI zone / presence classifiers.

Splits (all grouped so windows of one session never appear on both sides):
  session   stratified group k-fold over sessions
  day       leave-one-day-out
  temporal  forward chaining: train on days < D, test on day D (accuracy vs. days since training)
  room      leave-one-room-out
  leaky     random window k-fold. DELIBERATELY LEAKY, only to quantify how much a naive split inflates results.

Tasks: zone (empty/zoneA/...), presence (empty vs occupied: includes people standing still), or
motion (moving vs static, where static = empty + still: what motion detectors actually measure).
Baselines: threshold (ESPectre-style moving-variance threshold: P95 of empty training windows x 1.1,
presence only), logistic regression, random forest, small MLP. Scalers are fit on training folds only.

Usage:
  python csi_eval.py --data ../../data/raw --split day --task zone --models rf,lr,mlp
  python csi_eval.py --data ../../data/raw --suite --out ../../results/report.md
"""
import argparse
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found")

from csi_features import FeatureConfig, baseline_stats, session_windows
from csi_io import load_session, read_meta


@dataclass
class Table:
    X: np.ndarray
    y: np.ndarray
    session: np.ndarray
    day: np.ndarray
    activity: np.ndarray
    room: np.ndarray
    names: list


def discover(data_dir):
    return sorted(p for p in Path(data_dir).rglob("*.csv") if not p.stem.endswith("_check"))


def build_table(paths, cfg, baseline_ref=False):
    metas = {p: read_meta(p) for p in paths}
    base_by_day = {}
    if baseline_ref or cfg.n_select:
        grouped = defaultdict(list)
        for p, m in metas.items():
            if m.get("activity") == "baseline":
                grouped[m.get("day")].append(p)
        for day, ps in grouped.items():
            base_by_day[day] = baseline_stats([load_session(p) for p in ps], cfg)
    cols = defaultdict(list)
    X, names = [], None
    for p, m in metas.items():
        if m.get("activity") == "baseline" or m.get("label") is None:
            continue
        day = m.get("day")
        if (baseline_ref or cfg.n_select) and day not in base_by_day:
            raise SystemExit(f"day {day}: no baseline session (activity=baseline) for {p.name}")
        fp, sel = base_by_day.get(day, (None, None))
        Xs, _, names = session_windows(load_session(p), cfg, fp if baseline_ref else None, sel)
        if len(Xs) == 0:
            continue
        X.append(Xs)
        n = len(Xs)
        cols["y"] += [m["label"]] * n
        cols["session"] += [m["session"]] * n
        cols["day"] += [day] * n
        cols["activity"] += [m.get("activity")] * n
        cols["room"] += [m.get("room") or "room?"] * n
    if not X:
        raise SystemExit("no usable windows found")
    return Table(np.concatenate(X), *(np.array(cols[k]) for k in ("y", "session", "day", "activity", "room")), names)


# ---------------------------------------------------------------- models
class ThresholdDetector:
    """Moving-variance style baseline: score = turb_std^2, threshold = P95(negative training windows) * 1.1."""
    def __init__(self, idx, pos, neg):
        self.idx, self.pos, self.neg = idx, pos, neg

    def fit(self, X, y):
        s = X[:, self.idx] ** 2
        self.thr_ = np.percentile(s[y == self.neg], 95) * 1.1
        return self

    def predict(self, X):
        return np.where(X[:, self.idx] ** 2 > self.thr_, self.pos, self.neg)


def make_model(name, names, task="zone"):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, early_stopping=True,
                                                             max_iter=500, random_state=0))
    if name == "threshold":
        pos, neg = ("moving", "static") if task == "motion" else ("occupied", "empty")
        return ThresholdDetector(names.index("turb_std"), pos, neg)
    raise ValueError(name)


LEVEL = {"turb_mean", "turb_max", "turb_min"}   # track the static channel; drift from day to day


def feature_columns(names, group):
    """motion = temporal dynamics only; fp = per-subcarrier fingerprint; both = motion + fp;
    level = turbulence level only; all = everything (level features included)."""
    fp = [i for i, n in enumerate(names) if n.startswith("fp_")]
    lv = [i for i, n in enumerate(names) if n in LEVEL]
    mo = [i for i, n in enumerate(names) if not n.startswith("fp_") and n not in LEVEL]
    return {"motion": mo, "fp": fp, "both": mo + fp, "level": lv, "all": mo + lv + fp}[group]


# ---------------------------------------------------------------- splits
def make_splits(kind, T, y, n_splits=5, seed=0):
    idx = np.arange(len(y))
    if kind == "session":
        k = min(n_splits, len(set(T.session)))
        gen = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, y, T.session)
        return [(f"fold{i}", tr, te, {}) for i, (tr, te) in enumerate(gen)]
    if kind in ("day", "room"):
        g = T.day if kind == "day" else T.room
        if len(set(g)) < 2:
            raise SystemExit(f"split '{kind}' needs at least 2 distinct {kind}s in the data")
        return [(f"{kind}={g[te[0]]}", tr, te, {}) for tr, te in LeaveOneGroupOut().split(idx, y, g)]
    if kind == "temporal":
        days = sorted(set(T.day))
        if len(days) < 2:
            raise SystemExit("split 'temporal' needs at least 2 days")
        out = []
        for i in range(1, len(days)):
            tr, te = idx[np.isin(T.day, days[:i])], idx[T.day == days[i]]
            gap = (dt.date.fromisoformat(days[i]) - dt.date.fromisoformat(days[i - 1])).days
            out.append((f"train<{days[i]}", tr, te, {"gap_days": gap, "n_train_days": i}))
        return out
    if kind == "leaky":
        gen = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx, y)
        return [(f"fold{i}", tr, te, {}) for i, (tr, te) in enumerate(gen)]
    raise ValueError(kind)


def _check_no_leak(kind, T, tr, te):
    if kind != "leaky" and set(T.session[tr]) & set(T.session[te]):
        raise RuntimeError("LEAK: a session appears in both train and test")


# ---------------------------------------------------------------- metrics
def summarize(yt, yp, labels):
    bal = balanced_accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return {"bal_acc": bal, "macro_f1": f1, "precision": dict(zip(labels, p)), "recall": dict(zip(labels, r)),
            "cm": confusion_matrix(yt, yp, labels=labels)}


def bootstrap_ci(yt, yp, sessions, B=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(sessions)
    by = {s: np.flatnonzero(sessions == s) for s in ids}
    vals = []
    for _ in range(B):
        pick = np.concatenate([by[s] for s in rng.choice(ids, len(ids))])
        vals.append(balanced_accuracy_score(yt[pick], yp[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(T, kind="session", task="zone", models=("rf", "lr"), feats="both", seed=0):
    if task == "zone":
        y = T.y
    elif task == "presence":
        y = np.where(T.y == "empty", "empty", "occupied")
    else:
        y = np.where(T.activity == "moving", "moving", "static")
    labels = [str(c) for c in sorted(set(y), key=lambda c: (c != "empty", c))]
    cols = feature_columns(T.names, feats)
    splits = make_splits(kind, T, y, seed=seed)
    res = {}
    for mname in models:
        if mname == "threshold" and task == "zone":
            continue
        yt_all, yp_all, folds, trows = [], [], [], []
        idx_all = []
        for fname, tr, te, extra in splits:
            _check_no_leak(kind, T, tr, te)
            m = make_model(mname, T.names, task)
            if mname == "threshold":
                m.fit(T.X[tr], y[tr]); yp = m.predict(T.X[te])
            else:
                m.fit(T.X[tr][:, cols], y[tr]); yp = m.predict(T.X[te][:, cols])
            yt_all.append(y[te]); yp_all.append(yp); idx_all.append(te)
            s = summarize(y[te], yp, labels)
            folds.append((fname, s["bal_acc"], s["macro_f1"]))
            if extra:
                trows.append({**extra, "test": fname, "bal_acc": s["bal_acc"],
                              "acc": float(np.mean(y[te] == yp))})
        yt, yp, idx = np.concatenate(yt_all), np.concatenate(yp_all), np.concatenate(idx_all)
        pooled = summarize(yt, yp, labels)
        emp = T.y[idx] == "empty"
        pos_pred = (yp == "moving") if task == "motion" else (yp != "empty")
        by_act = {a: float(np.mean(yp[(T.activity[idx] == a)] == yt[(T.activity[idx] == a)]))
                  for a in sorted(set(T.activity[idx])) if a not in ("empty",)}
        fb = np.array([f[1] for f in folds])
        res[mname] = {"pooled": pooled, "folds": folds, "fold_mean": float(fb.mean()), "fold_std": float(fb.std()),
                      "fpr_empty": float(np.mean(pos_pred[emp])) if emp.any() else float("nan"),
                      "recall_by_activity": by_act, "temporal": trows,
                      "ci": bootstrap_ci(yt, yp, T.session[idx]), "labels": labels}
    return res


def format_results(res, title):
    out = [f"### {title}", ""]
    out += ["| model | balanced acc (pooled) | 95% CI (session bootstrap) | folds mean ± std | macro F1 | FPR on empty |",
            "|---|---|---|---|---|---|"]
    for m, r in res.items():
        p = r["pooled"]
        out.append(f"| {m} | {p['bal_acc']:.3f} | [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | "
                   f"{r['fold_mean']:.3f} ± {r['fold_std']:.3f} | {p['macro_f1']:.3f} | {r['fpr_empty']:.3f} |")
    for m, r in res.items():
        out += ["", f"**{m}** per-class recall: " + ", ".join(f"{k}={v:.2f}" for k, v in r["pooled"]["recall"].items())
                + " | accuracy by activity: " + ", ".join(f"{k}={v:.2f}" for k, v in r["recall_by_activity"].items())]
        if r["temporal"]:
            out += ["", f"{m} accuracy vs. time since training:", "", "| test day | train days | gap (days) | acc |", "|---|---|---|---|"]
            out += [f"| {t['test']} | {t['n_train_days']} | {t['gap_days']} | {t['acc']:.3f} |" for t in r["temporal"]]
        labs = r["labels"]
        out += ["", f"confusion matrix ({m}; rows=true, cols=pred; order {labs}):", "", "```"]
        out += [" ".join(f"{v:5d}" for v in row) for row in r["pooled"]["cm"]] + ["```"]
    return "\n".join(out)


def run_suite(paths, cfg, out):
    sections = ["# CSI evaluation report", "",
                "Synthetic or real data: check the data folder before quoting any number from this file.", ""]
    have_base = any(read_meta(p).get("activity") == "baseline" for p in paths)
    variants = [("raw fingerprints", False)] + ([("baseline-referenced fingerprints", True)] if have_base else [])
    for vname, bref in variants:
        T = build_table(paths, cfg, baseline_ref=bref)
        sections.append(f"## Features: {vname} (windows={len(T.y)}, sessions={len(set(T.session))}, days={sorted(set(T.day))})\n")
        for task in ("motion", "presence", "zone"):
            for kind in ("session", "day", "temporal", "leaky"):
                if kind in ("day", "temporal") and len(set(T.day)) < 2:
                    continue
                for feats in ("motion", "fp", "both", "all"):
                    models = ("threshold", "rf", "lr", "mlp") if (task != "zone" and feats == "motion") else ("rf", "lr")
                    print(f"  {vname}: {task} / {kind} / {feats}", flush=True)
                    try:
                        r = evaluate(T, kind, task, models, feats)
                    except Exception as e:   # e.g. a fold with a single class
                        sections.append(f"*{task}/{kind}/{feats}: skipped ({e})*\n")
                        continue
                    flag = " (LEAKY: random window split, for comparison only)" if kind == "leaky" else ""
                    sections.append(format_results(r, f"{task} | split={kind}{flag} | features={feats}") + "\n")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(sections))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="session", choices=["session", "day", "temporal", "room", "leaky"])
    ap.add_argument("--task", default="zone", choices=["zone", "presence", "motion"])
    ap.add_argument("--models", default="rf,lr")
    ap.add_argument("--features", default="both", choices=["motion", "fp", "both", "level", "all"])
    ap.add_argument("--baseline-ref", action="store_true")
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--win-s", type=float, default=5.0)
    ap.add_argument("--step-s", type=float, default=2.5)
    ap.add_argument("--n-select", type=int, default=0)
    ap.add_argument("--turb", default="std", choices=["std", "cv"])
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--out", default="../../results/report.md")
    a = ap.parse_args()
    cfg = FeatureConfig(fs=a.fs, win_s=a.win_s, step_s=a.step_s, n_select=a.n_select, turb_mode=a.turb)
    paths = discover(a.data)
    if a.suite:
        return run_suite(paths, cfg, a.out)
    T = build_table(paths, cfg, a.baseline_ref)
    res = evaluate(T, a.split, a.task, a.models.split(","), a.features)
    print(format_results(res, f"{a.task} | split={a.split} | features={a.features}"))


if __name__ == "__main__":
    main()
