"""Realism checks on decoded books, each judged against real data.

The checks follow the tutor's list:
    1  the book at one time       spread and level-1 queue imbalance
                                  (rule violations are counted in metrics.py)
    2  how bids and asks move     changes of the bid and ask VWAP over 1, 10 and 50 steps

Every check is a distance d between two sets of windows, in ticks where it
has a unit. Wasserstein-1 replaces binned KL: it needs no bins and no
smoothing, and reads as "how many ticks off on average". The KS statistic is
the largest gap between the two cumulative distributions. Its p-value is not
used, because steps inside a window are not independent.

A model is judged against two real yardsticks. Each compares n_test windows
with the real training windows, where n_test is the number of
non-overlapping test windows:
    F95  noise floor: the 95th percentile of d(n_test random training windows,
         the other training windows) over N_DRAWS draws
    D    drift: d(real test windows, real training windows), how far a later
         real hour already is
    M    model: the median of d(n_test random synthetic windows, real training
         windows) over N_DRAWS draws, so all three use the same sample size
A check passes when M <= max(D, F95), meaning the synthetic data is no
further from training than a later real hour, and is indistinguishable when
M <= F95. The KL and SSIM targets of the spec are reported separately in
audit.py.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from dataset import TICK, OrderBook
from helpers.metrics import side_vwap

N_DRAWS = 200
HORIZONS = (1, 10, 50)

Features = dict[str, np.ndarray]                # per-window arrays, first axis = window
Distance = Callable[[Features, np.ndarray, Features, np.ndarray], float]


# ---------------------------------------------------------------------------
# Per-window series
# ---------------------------------------------------------------------------

def _numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().double().numpy()


def queue_imbalance(book: OrderBook) -> torch.Tensor:
    """(bid size - ask size) / (bid size + ask size) at level 1, in [-1, 1] (Gould and Bonart, 2016)."""
    bid, ask = book.bid_size[:, 0].clamp_min(0), book.ask_size[:, 0].clamp_min(0)
    return (bid - ask) / (bid + ask).clamp_min(1e-9)


def window_features(books: list[OrderBook]) -> Features:
    """The series every check reads, stacked per window.

    spread           (n, T)          spread in ticks
    imbalance        (n, T)          level-1 queue imbalance
    vwap             (n, 2, T)       bid and ask VWAP in ticks
    vwap_change_<h>  (n, 2, T - h)   VWAP change over h steps in ticks
    """
    vwap = np.stack([np.stack([_numpy(side_vwap(b.bid_price, b.bid_size)),
                               _numpy(side_vwap(b.ask_price, b.ask_size))]) for b in books]) / TICK
    features = {
        "spread": np.stack([np.round(_numpy(b.spread) / TICK) for b in books]),
        "imbalance": np.stack([_numpy(queue_imbalance(b)) for b in books]),
        "vwap": vwap,
    }
    for h in HORIZONS:
        if h < vwap.shape[2]:
            features[f"vwap_change_{h}"] = vwap[:, :, h:] - vwap[:, :, :-h]
    return features


# ---------------------------------------------------------------------------
# Distances between two sets of windows
# ---------------------------------------------------------------------------

def wasserstein(key: str) -> Distance:
    """W1 between the pooled values of one series."""
    def d(a: Features, ia: np.ndarray, b: Features, ib: np.ndarray) -> float:
        return float(wasserstein_distance(a[key][ia].ravel(), b[key][ib].ravel()))
    return d


def ks_statistic(key: str) -> Distance:
    """Largest gap between the two empirical CDFs of one series."""
    def d(a: Features, ia: np.ndarray, b: Features, ib: np.ndarray) -> float:
        return float(ks_2samp(a[key][ia].ravel(), b[key][ib].ravel(), method="asymp").statistic)
    return d


def checks(features: Features) -> dict[str, tuple[int, str, Distance]]:
    """Every check that applies to windows of this length: name -> (tutor point, unit, distance)."""
    out = {
        "spread_w1": (1, "ticks", wasserstein("spread")),
        "spread_ks": (1, "KS", ks_statistic("spread")),
        "imbalance_w1": (1, "imbalance", wasserstein("imbalance")),
    }
    for h in HORIZONS:
        if f"vwap_change_{h}" in features:
            out[f"vwap_change_{h}_w1"] = (2, "ticks", wasserstein(f"vwap_change_{h}"))
    out["vwap_change_1_ks"] = (2, "KS", ks_statistic("vwap_change_1"))
    return out


# ---------------------------------------------------------------------------
# Judging a model against the real yardsticks
# ---------------------------------------------------------------------------

def judge(train: Features, test: Features, synthetic: Features, draws: int = N_DRAWS,
          seed: int = 0) -> dict[str, dict]:
    """F95, D and M for every check, with a verdict (see the module docstring).

    train and test must hold non-overlapping windows. model_all and
    model_vs_test use every synthetic window, against training and against
    the test hour.
    """
    n_train, n_test, n_fake = (len(f["spread"]) for f in (train, test, synthetic))
    if n_test >= n_train:
        raise ValueError("The noise floor needs more training windows than test windows")
    rng = np.random.default_rng(seed)
    splits = [rng.permutation(n_train) for _ in range(draws)]
    picks = [rng.choice(n_fake, size=n_test, replace=False) for _ in range(draws)]
    every_train, every_test, every_fake = np.arange(n_train), np.arange(n_test), np.arange(n_fake)

    report = {}
    for name, (point, unit, d) in checks(train).items():
        floor = np.array([d(train, s[:n_test], train, s[n_test:]) for s in splits])
        model = np.array([d(synthetic, p, train, every_train) for p in picks])
        drift = d(test, every_test, train, every_train)
        m, f95 = float(np.median(model)), float(np.quantile(floor, 0.95))
        report[name] = {
            "point": point, "unit": unit,
            "floor_median": float(np.median(floor)), "floor_95": f95, "drift": drift, "model": m,
            "model_all": d(synthetic, every_fake, train, every_train),
            "model_vs_test": d(synthetic, every_fake, test, every_test),
            "verdict": "indistinguishable" if m <= f95 else "pass" if m <= max(drift, f95) else "fail",
        }
    return report


def realism_table(report: dict[str, dict], title: str) -> str:
    """Markdown table of one judge() report."""
    lines = [f"**{title}**", "",
             "| Check | Point | Unit | Floor F95 | Drift D | Model M | Verdict | All synthetic vs test |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in report.items():
        lines.append(f"| {name} | {r['point']} | {r['unit']} | {r['floor_95']:.3f} | {r['drift']:.3f} | "
                     f"{r['model']:.3f} | {r['verdict']} | {r['model_vs_test']:.3f} |")
    return "\n".join(lines) + "\n"
