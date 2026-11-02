"""
Feature extraction for LLTF CSI amplitude streams.

Pipeline per session:
  LLTF amplitude (valid subcarriers) -> uniform resampling (gaps flagged) ->
  per-packet "turbulence" (spread across subcarriers) -> Hampel outlier filter ->
  sliding windows -> window features.

Window features
  turbulence statistics (9): mean, std, max, min, IQR, MAD, skew, lag-1 autocorrelation,
      waveform length; plus 3 level-invariant ratios (std, IQR, waveform length over mean).
      mean/max/min are "level" features: they track the static channel, which changes from day to day,
      so they are kept out of the default feature groups (see csi_eval.feature_columns).
  mean_cv: mean over subcarriers of (std/mean over time)
  corr_lag_*: cosine similarity of the mean-removed amplitude profile between packets
      `lag_s` apart (1 = unchanged channel shape, lower = body motion)
  bp_turb_*, bp_amp_*: log band power of the turbulence series and of the mean amplitude
  fp_k: normalised mean amplitude per subcarrier (static fingerprint). With a per-day empty-room
      baseline it becomes log(fp / baseline_fp), which removes slow environment drift.

Subcarrier selection (optional) re-implements the baseline-stability score documented by the
ESPectre project (alpha*sigma/mu^2 + (1-alpha)*sigma/mu, lowest = most stable) from its written
description. Note this is NOT the same quantity as `mean_cv` (an earlier draft of this repo
mislabelled mean_cv as NBVI).
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.stats import skew

LO_POS = np.arange(1, 27)     # +1..+26
HI_POS = np.arange(38, 64)    # -26..-1


def valid_positions(edge_trim: int = 2) -> np.ndarray:
    """LLTF buffer positions that carry signal, minus `edge_trim` weak band-edge bins per side."""
    return np.concatenate([LO_POS[: len(LO_POS) - edge_trim], HI_POS[edge_trim:]])


@dataclass
class FeatureConfig:
    fs: float = 50.0            # resampling rate (Hz); firmware sends ~100 pkt/s
    win_s: float = 5.0
    step_s: float = 2.5
    edge_trim: int = 2
    hampel_win: int = 7
    hampel_thr: float = 5.0
    turb_mode: str = "std"      # "std" (raw spread) or "cv" (gain-invariant spread/mean)
    n_select: int = 0           # >0: keep this many baseline-stable subcarriers for turbulence features
    max_gap_s: float = 0.25     # windows containing a longer packet gap are dropped
    lag_s: float = 0.1
    fingerprint: bool = True


def resample_uniform(t, X, fs, max_gap_s):
    """Linear resample onto a uniform grid. Returns (grid, X_grid, ok) where ok marks grid
    points that are not inside a packet gap longer than max_gap_s."""
    g = np.arange(t[0], t[-1], 1.0 / fs)
    Xg = interp1d(t, X, axis=0, kind="linear", assume_sorted=True)(g)
    idx = np.clip(np.searchsorted(t, g, side="right"), 1, len(t) - 1)
    ok = (t[idx] - t[idx - 1]) <= max_gap_s
    return g, Xg, ok


def hampel(x, win=7, thr=5.0):
    """Replace outliers (|x - median| > thr * 1.4826 * MAD over a centred window) with the median."""
    if win < 3:
        return x.copy()
    win += (win % 2 == 0)
    h = win // 2
    W = np.lib.stride_tricks.sliding_window_view(np.pad(x, h, mode="edge"), win)
    med = np.median(W, axis=1)
    mad = np.median(np.abs(W - med[:, None]), axis=1)
    out = x.copy()
    bad = (mad > 1e-9) & (np.abs(x - med) > thr * 1.4826 * mad)
    out[bad] = med[bad]
    return out


def turbulence(A, mode="std"):
    sd = A.std(axis=1)
    if mode == "cv":
        return sd / (A.mean(axis=1) + 1e-9)
    return sd


def nbvi_select(A, k, alpha=0.75, noise_gate_pct=15.0):
    """Pick the k most baseline-stable subcarriers (columns of A, an empty-room recording)."""
    mu, sd = A.mean(0), A.std(0)
    score = alpha * sd / (mu ** 2 + 1e-12) + (1 - alpha) * sd / (mu + 1e-12)
    score = np.where(mu >= np.percentile(mu, noise_gate_pct), score, np.inf)
    return np.sort(np.argsort(score)[:k])


def amplitude_valid(sess, cfg):
    return np.abs(sess.lltf[:, valid_positions(cfg.edge_trim)])


def baseline_stats(sessions, cfg):
    """From a day's empty-room baseline sessions: normalised fingerprint and (optionally) the
    stable-subcarrier selection."""
    fps, As = [], []
    for s in sessions:
        A = amplitude_valid(s, cfg)
        m = A.mean(0)
        fps.append(m / m.mean())
        As.append(A)
    fp = np.mean(fps, axis=0)
    sel = nbvi_select(np.concatenate(As), cfg.n_select) if cfg.n_select else None
    return fp, sel


def _band_logpow(x, fs, bands):
    f, p = welch(x, fs=fs, nperseg=min(len(x), 1024), detrend="constant")
    df = f[1] - f[0] if len(f) > 1 else 1.0
    return [float(np.log10(p[(f >= lo) & (f < hi)].sum() * df + 1e-12)) for lo, hi in bands]


def feature_names(cfg, n_fp):
    names = ["turb_mean", "turb_std", "turb_max", "turb_min", "turb_iqr", "turb_mad",
             "turb_skew", "turb_ac1", "turb_wl", "turb_rel_std", "turb_rel_iqr", "turb_rel_wl",
             "mean_cv", "corr_lag_mean", "corr_lag_min"]
    for b in _bands(cfg):
        names.append(f"bp_turb_{b[0]}_{b[1]}")
    for b in _bands(cfg):
        names.append(f"bp_amp_{b[0]}_{b[1]}")
    if cfg.fingerprint:
        names += [f"fp_{i}" for i in range(n_fp)]
    return names


def _bands(cfg):
    bands = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)]
    if cfg.win_s >= 10:
        bands = [(0.1, 0.5)] + bands      # breathing band is only resolvable in long windows
    return bands


def window_vector(turb_w, A_sel_w, A_all_w, cfg, base_fp):
    fs = cfg.fs
    x = turb_w
    mean, std = x.mean(), x.std()
    q75, q25 = np.percentile(x, [75, 25])
    xc = x - mean
    den = float((xc ** 2).sum())
    v = [mean, std, x.max(), x.min(), q75 - q25, np.median(np.abs(x - np.median(x))),
         float(skew(x)) if std > 1e-12 else 0.0,
         float((xc[:-1] * xc[1:]).sum() / den) if den > 1e-12 else 0.0,
         float(np.abs(np.diff(x)).sum())]
    lvl = abs(mean) + 1e-9          # level-invariant versions of the dynamics
    v += [std / lvl, (q75 - q25) / lvl, v[8] / (lvl * (len(x) - 1))]
    mu = A_sel_w.mean(0)
    v.append(float(np.mean(A_sel_w.std(0) / (mu + 1e-9))))
    P = A_sel_w - A_sel_w.mean(1, keepdims=True)
    L = max(1, int(round(cfg.lag_s * fs)))
    num = (P[:-L] * P[L:]).sum(1)
    den2 = np.linalg.norm(P[:-L], axis=1) * np.linalg.norm(P[L:], axis=1) + 1e-9
    sim = num / den2
    v += [float(sim.mean()), float(sim.min())]
    v += _band_logpow(x, fs, _bands(cfg))
    v += _band_logpow(A_sel_w.mean(1), fs, _bands(cfg))
    if cfg.fingerprint:
        fp = A_all_w.mean(0)
        fp = fp / fp.mean()
        if base_fp is not None:
            fp = np.log(fp / base_fp)
        v += fp.tolist()
    return np.asarray(v, dtype=np.float64)


def session_windows(sess, cfg, base_fp=None, sel=None):
    """Return (X, t_start, names) for all complete windows of one session."""
    A_all = amplitude_valid(sess, cfg)
    g, Ag, ok = resample_uniform(sess.t, A_all, cfg.fs, cfg.max_gap_s)
    As = Ag[:, sel] if sel is not None else Ag
    turb = hampel(turbulence(As, cfg.turb_mode), cfg.hampel_win, cfg.hampel_thr)
    W, S = int(cfg.win_s * cfg.fs), int(cfg.step_s * cfg.fs)
    X, ts = [], []
    for a in range(0, len(g) - W + 1, S):
        if not ok[a:a + W].all():
            continue
        X.append(window_vector(turb[a:a + W], As[a:a + W], Ag[a:a + W], cfg, base_fp))
        ts.append(g[a])
    names = feature_names(cfg, A_all.shape[1])
    if not X:
        return np.zeros((0, len(names))), np.zeros(0), names
    return np.stack(X), np.asarray(ts), names
