"""
Feature extraction for CSI amplitude streams.
Implements NBVI (Normalized Baseline Variability Index) over a sliding window
of packets, the core feature used in recent WiFi CSI occupancy/motion literature.
"""
import numpy as np

VALID_SUBCARRIER_RANGE = (6, 122)  # trims guard/pilot/zero-padded subcarriers; tune after real data inspection

def trim_subcarriers(amplitude: np.ndarray) -> np.ndarray:
    """Drop guard-band and zero-padded subcarriers, keep only the informative range."""
    lo, hi = VALID_SUBCARRIER_RANGE
    hi = min(hi, len(amplitude))
    return amplitude[lo:hi]

def nbvi(amplitude_window: np.ndarray) -> float:
    """
    Normalized Baseline Variability Index over a window of packets.
    amplitude_window: shape (W, K) — W packets, K subcarriers.
    NBVI = (1/K) * sum_k ( std(amplitude[:,k]) / mean(amplitude[:,k]) )
    Higher NBVI = more relative fluctuation across subcarriers = more likely motion/occupancy.
    """
    mu = amplitude_window.mean(axis=0)
    sigma = amplitude_window.std(axis=0)
    mask = mu > 1e-6
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(sigma[mask] / mu[mask]))

def sliding_windows(packets: list[np.ndarray], window_size: int = 1000, step: int = 500):
    """
    Yield (start_idx, window_array) tuples from a list of per-packet amplitude arrays.
    window_size=1000, step=500: a 10s window with 50% overlap, at the firmware's confirmed
    TX rate of 100 packets/sec (CONFIG_SEND_FREQUENCY in csi_send/main/*.c).
    If CONFIG_SEND_FREQUENCY is changed in firmware, recompute: window_size = send_freq_hz * target_window_s.
    """
    for start in range(0, len(packets) - window_size + 1, step):
        window = np.stack(packets[start:start + window_size])
        yield start, window
