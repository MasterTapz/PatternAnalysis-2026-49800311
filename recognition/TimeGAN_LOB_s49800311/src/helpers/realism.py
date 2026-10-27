"""Realism checks on decoded books, each judged against real data.

The checks follow the tutor's list:
    1  the book at one time       spread and level-1 queue imbalance
                                  (rule violations are counted in metrics.py)
    2  how bids and asks move     changes of the bid and ask VWAP over 1, 10 and 50 steps
    4  heteroskedasticity         ACF of |mid returns| and |VWAP changes|, the share of
                                  windows where Engle's ARCH-LM test finds volatility
                                  clustering, and how much volatility differs between windows

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
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import chi2, ks_2samp, wasserstein_distance

from dataset import TICK, OrderBook
from helpers.metrics import side_vwap

N_DRAWS = 200
HORIZONS = (1, 10, 50)
ACF_LAGS = 20
ARCH_LAGS = 5
ARCH_ALPHA = 0.05

Features = dict[str, np.ndarray]                # per-window arrays, first axis = window
Distance = Callable[[Features, np.ndarray, Features, np.ndarray], float]


@dataclass
class Check:
    """One realism check: its tutor point, unit, distance and an optional summary value of one set."""

    point: int
    unit: str
    distance: Distance
    value: Callable[[Features], float] | None = None


# ---------------------------------------------------------------------------
# Per-window series
# ---------------------------------------------------------------------------

def _numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().double().numpy()


def queue_imbalance(book: OrderBook) -> torch.Tensor:
    """(bid size - ask size) / (bid size + ask size) at level 1, in [-1, 1] (Gould and Bonart, 2016)."""
    bid, ask = book.bid_size[:, 0].clamp_min(0), book.ask_size[:, 0].clamp_min(0)
    return (bid - ask) / (bid + ask).clamp_min(1e-9)


def window_acf(x: np.ndarray, max_lag: int = ACF_LAGS) -> np.ndarray:
    """ACF at lags 1..max_lag of series (k, T), pooled over windows.

    Each window is demeaned on its own, so windows that are calmer or busier
    than others do not add correlation at every lag; no pair crosses a
    window edge. A set with no variation at all gets zeros.
    """
    x = x - x.mean(axis=1, keepdims=True)
    var = (x * x).sum()
    if var == 0:
        return np.zeros(max_lag)
    return np.array([(x[:, :-k] * x[:, k:]).sum() / var for k in range(1, max_lag + 1)])


def arch_lm_rejects(returns: np.ndarray, lags: int = ARCH_LAGS, alpha: float = ARCH_ALPHA) -> np.ndarray:
    """Engle's (1982) ARCH-LM test on each window of returns (n, T), True where it rejects.

    The residuals of an AR(1) fit remove the bid-ask bounce. Their squares are
    regressed on their own last `lags` values, and T * R^2 is compared with
    chi-squared(lags). A window whose squared residuals never vary cannot
    show clustering and counts as not rejected.
    """
    rejects = np.zeros(len(returns), dtype=bool)
    for i, r in enumerate(returns):
        ar = np.column_stack([np.ones(len(r) - 1), r[:-1]])
        residual = r[1:] - ar @ np.linalg.lstsq(ar, r[1:], rcond=None)[0]
        e2 = residual ** 2
        y = e2[lags:]
        if y.var() == 0:
            continue
        x = np.column_stack([np.ones(len(y))] + [e2[lags - k:-k] for k in range(1, lags + 1)])
        fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rejects[i] = chi2.sf(len(y) * r2, lags) < alpha
    return rejects


def window_features(books: list[OrderBook]) -> Features:
    """The series every check reads, stacked per window.

    spread           (n, T)          spread in ticks
    imbalance        (n, T)          level-1 queue imbalance
    vwap             (n, 2, T)       bid and ask VWAP in ticks
    vwap_change_<h>  (n, 2, T - h)   VWAP change over h steps in ticks
    returns          (n, T - 1)      mid log-returns
    arch_reject      (n,)            the ARCH-LM test rejects on the returns
    window_vol       (n,)            std of the 1-step VWAP change (bid and ask averaged)
    """
    vwap = np.stack([np.stack([_numpy(side_vwap(b.bid_price, b.bid_size)),
                               _numpy(side_vwap(b.ask_price, b.ask_size))]) for b in books]) / TICK
    returns = np.stack([np.diff(np.log(_numpy(b.mid))) for b in books])
    features = {
        "spread": np.stack([np.round(_numpy(b.spread) / TICK) for b in books]),
        "imbalance": np.stack([_numpy(queue_imbalance(b)) for b in books]),
        "vwap": vwap,
        "returns": returns,
        "arch_reject": arch_lm_rejects(returns),
    }
    for h in HORIZONS:
        if h < vwap.shape[2]:
            features[f"vwap_change_{h}"] = vwap[:, :, h:] - vwap[:, :, :-h]
    features["window_vol"] = features["vwap_change_1"].std(axis=2).mean(axis=1)
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


def _abs_series(features: Features, key: str, idx: np.ndarray) -> np.ndarray:
    """|series| of the chosen windows as (k, T); bid and ask VWAP rows count as separate series."""
    x = np.abs(features[key][idx])
    return x.reshape(-1, x.shape[-1])


def acf_gap(key: str) -> Distance:
    """L2 distance between the ACFs of |series| at lags 1..ACF_LAGS (Wiese et al., 2020).

    Absolute values rather than squares: with excess kurtosis of 15 to 40,
    sample ACFs of squared returns are unstable (Cont, 2001).
    """
    def d(a: Features, ia: np.ndarray, b: Features, ib: np.ndarray) -> float:
        return float(np.linalg.norm(window_acf(_abs_series(a, key, ia)) - window_acf(_abs_series(b, key, ib))))
    return d


def statistic_gap(key: str, statistic: Callable[[np.ndarray], float]) -> Distance:
    """|statistic(set a) - statistic(set b)| of a per-window array."""
    def d(a: Features, ia: np.ndarray, b: Features, ib: np.ndarray) -> float:
        return abs(float(statistic(a[key][ia])) - float(statistic(b[key][ib])))
    return d


def checks(features: Features) -> dict[str, Check]:
    """Every check that applies to windows of this length."""
    every = slice(None)
    out = {
        "spread_w1": Check(1, "ticks", wasserstein("spread")),
        "spread_ks": Check(1, "KS", ks_statistic("spread")),
        "imbalance_w1": Check(1, "imbalance", wasserstein("imbalance")),
    }
    for h in HORIZONS:
        if f"vwap_change_{h}" in features:
            out[f"vwap_change_{h}_w1"] = Check(2, "ticks", wasserstein(f"vwap_change_{h}"))
    out["vwap_change_1_ks"] = Check(2, "KS", ks_statistic("vwap_change_1"))
    out |= {
        "acf_abs_return_l2": Check(4, "L2, lags 1-20", acf_gap("returns"),
                                   lambda f: window_acf(_abs_series(f, "returns", every))[0]),
        "acf_abs_vwap_change_l2": Check(4, "L2, lags 1-20", acf_gap("vwap_change_1"),
                                        lambda f: window_acf(_abs_series(f, "vwap_change_1", every))[0]),
        "arch_lm_share": Check(4, "share", statistic_gap("arch_reject", np.mean),
                               lambda f: float(f["arch_reject"].mean())),
        "window_vol_sd": Check(4, "ticks", statistic_gap("window_vol", np.std),
                               lambda f: float(f["window_vol"].std())),
    }
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
    for name, check in checks(train).items():
        d = check.distance
        floor = np.array([d(train, s[:n_test], train, s[n_test:]) for s in splits])
        model = np.array([d(synthetic, p, train, every_train) for p in picks])
        drift = d(test, every_test, train, every_train)
        m, f95 = float(np.median(model)), float(np.quantile(floor, 0.95))
        report[name] = {
            "point": check.point, "unit": check.unit,
            "floor_median": float(np.median(floor)), "floor_95": f95, "drift": drift, "model": m,
            "model_all": d(synthetic, every_fake, train, every_train),
            "model_vs_test": d(synthetic, every_fake, test, every_test),
            "verdict": "indistinguishable" if m <= f95 else "pass" if m <= max(drift, f95) else "fail",
        }
        if check.value is not None:
            report[name]["values"] = {k: float(check.value(f)) for k, f in
                                      [("real_train", train), ("real_test", test), ("synthetic", synthetic)]}
    return report


def realism_table(report: dict[str, dict], title: str) -> str:
    """Markdown table of one judge() report.

    The last column gives the value behind a check on real training, real
    test and synthetic windows: the lag-1 ACF for ACF checks, the statistic itself otherwise.
    """
    lines = [f"**{title}**", "",
             "| Check | Point | Unit | Floor F95 | Drift D | Model M | Verdict | All synthetic vs test | "
             "Value: train / test / synthetic |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in report.items():
        values = " / ".join(f"{v:.3f}" for v in r["values"].values()) if "values" in r else ""
        lines.append(f"| {name} | {r['point']} | {r['unit']} | {r['floor_95']:.3f} | {r['drift']:.3f} | "
                     f"{r['model']:.3f} | {r['verdict']} | {r['model_vs_test']:.3f} | {values} |")
    return "\n".join(lines) + "\n"
