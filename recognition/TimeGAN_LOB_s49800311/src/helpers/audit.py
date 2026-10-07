"""The test-hour audit behind predict.py.

audit_run samples a trained model and compares its decoded books with the
real test hour, using real training windows as a reference for what is
achievable:
    invalid books   crossed or locked, broken ladder, negative size
    KL              spreads (ticks) and mid moves (half-ticks), target <= 0.1
    SSIM            depth heatmaps (20 levels x time, log volume), target > 0.6
    volatility      autocorrelation of squared returns, lags 1 to 20
    price path      how often the mid stays put and how far it wanders
    diversity       spread of samples and distance to the closest training window
NumPy and scikit-image are used for SSIM, which the spec allows here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset import TICK, LOBSplits, OrderBook
from helpers.metrics import (KL_TARGET, book_violations, decode_real_windows, decode_synthetic_windows,
                             distribution_report, kl_only, mid_moves, step_violations)
from modules import SequenceGAN

SSIM_TARGET = 0.6         # spec target for SSIM of depth snapshots
SSIM_RANDOM_PAIRS = 1024  # pairs drawn for the random-pair SSIM
ACF_LAGS = 20
ACF_KEY = "acf_squared_returns_lags_1_5_10_20"


# ---------------------------------------------------------------------------
# Depth heatmaps and SSIM
# ---------------------------------------------------------------------------

def depth_image(book: OrderBook) -> np.ndarray:
    """(20, T) heatmap of log1p(size), rows bid 10..1 then ask 1..10.

    Price rises up the image. Negative sizes are clipped to zero here; they
    are counted separately.
    """
    bid = torch.log1p(book.bid_size.clamp_min(0)).flip(1)
    ask = torch.log1p(book.ask_size.clamp_min(0))
    return torch.cat([bid, ask], dim=1).T.numpy()


def depth_data_range(splits: LOBSplits) -> float:
    """The largest log volume seen in training, used as the SSIM range for every heatmap."""
    size_cols = [i for i, name in enumerate(splits.feature_names) if "size" in name]
    return float((splits.scaler.shift + splits.scaler.scale)[size_cols].max())


def ssim(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """SSIM (Wang et al., 2004) with a 7x7 window."""
    return float(structural_similarity(a, b, data_range=data_range, win_size=7))


def _mean_pair_ssim(left: list[np.ndarray], right: list[np.ndarray], left_idx: np.ndarray,
                    right_idx: np.ndarray, data_range: float) -> float:
    return float(np.mean([ssim(left[a], right[b], data_range) for a, b in zip(left_idx, right_idx)]))


def _ssim_matrix(rows: list[np.ndarray], cols: list[np.ndarray], data_range: float) -> np.ndarray:
    return np.array([[ssim(r, c, data_range) for c in cols] for r in rows])


def ssim_scores(real_imgs: list[np.ndarray], fake_imgs: list[np.ndarray], ref_imgs: list[np.ndarray],
                data_range: float, rng: np.random.Generator) -> tuple[dict[str, float], np.ndarray]:
    """SSIM under three protocols, each with a real reference (training windows vs the test hour).

    random_pairs  mean over random (real test, synthetic) pairs
    nearest       per real test window, the best match in the synthetic pool
    mean_surface  the average synthetic surface against the average real one

    Even two real windows only reach about 0.1 under random pairing, so the
    0.6 target is only reachable for the mean surface. Also returns the
    (n_real, n_fake) SSIM matrix used for nearest.
    """
    real_idx = rng.integers(len(real_imgs), size=SSIM_RANDOM_PAIRS)
    fake_idx = rng.integers(len(fake_imgs), size=SSIM_RANDOM_PAIRS)
    ref_idx = rng.integers(len(ref_imgs), size=SSIM_RANDOM_PAIRS)
    nearest = _ssim_matrix(real_imgs, fake_imgs, data_range)
    nearest_ref = _ssim_matrix(real_imgs, ref_imgs, data_range)
    real_mean = np.mean(real_imgs, axis=0)
    scores = {
        "random_pairs": _mean_pair_ssim(real_imgs, fake_imgs, real_idx, fake_idx, data_range),
        "random_pairs_real_reference": _mean_pair_ssim(real_imgs, ref_imgs, real_idx, ref_idx, data_range),
        "nearest": float(nearest.max(axis=1).mean()),
        "nearest_real_reference": float(nearest_ref.max(axis=1).mean()),
        "mean_surface": ssim(np.mean(fake_imgs, axis=0), real_mean, data_range),
        "mean_surface_real_reference": ssim(np.mean(ref_imgs, axis=0), real_mean, data_range),
    }
    return scores, nearest


# ---------------------------------------------------------------------------
# Volatility clustering, price paths and diversity
# ---------------------------------------------------------------------------

def log_returns(books: list[OrderBook]) -> torch.Tensor:
    """Mid log-returns inside each window, shape (n_books, T - 1)."""
    return torch.stack([torch.diff(torch.log(b.mid)) for b in books])


def pooled_acf(x: torch.Tensor, max_lag: int = ACF_LAGS) -> np.ndarray:
    """Autocorrelation at lags 1..max_lag pooled over windows; no pair crosses a window edge."""
    x = x - x.mean()
    var = (x * x).mean()
    return np.array([float((x[:, :-k] * x[:, k:]).mean() / var) for k in range(1, max_lag + 1)])


def price_path_stats(books: list[OrderBook]) -> dict[str, float]:
    """How often the mid stays put and how far it wanders.

    zero_move_share    share of steps with no mid move (real quotes hold flat)
    excursion_ticks_*  per window, the furthest the mid gets from its first
                       value in ticks; median and 90th percentile over windows
    """
    moves = mid_moves(books)
    excursion = torch.stack([(b.mid - b.mid[0]).abs().max() / TICK for b in books]).double()
    return {
        "zero_move_share": float((moves == 0).double().mean()),
        "excursion_ticks_median": float(torch.quantile(excursion, 0.5)),   # median() would not interpolate
        "excursion_ticks_p90": float(torch.quantile(excursion, 0.9)),
    }


def mean_pairwise_distance(x: torch.Tensor, n: int = 512, seed: int = 0) -> float:
    """Average RMS distance between windows (B, T, F); low means low diversity."""
    g = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(len(x), generator=g)[:n]].flatten(1).double()
    d = torch.cdist(x, x) / np.sqrt(x.shape[1])
    return float(d[~torch.eye(len(x), dtype=torch.bool)].mean())


def nearest_train_distance(x: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    """RMS distance from each window in x to its closest training window."""
    a, b = x.flatten(1).double(), train.flatten(1).double()
    return torch.cdist(a, b).min(dim=1).values / np.sqrt(a.shape[1])


# ---------------------------------------------------------------------------
# Autopsy cases
# ---------------------------------------------------------------------------

def window_stats(books: list[OrderBook]) -> dict[str, np.ndarray]:
    """Per-window counts used to pick the autopsy cases."""
    masks = [step_violations(b) for b in books]
    ladder = np.array([int(m["ladder"].sum()) for m in masks])
    crossed = np.array([int(m["crossed"].sum()) for m in masks])
    negative = np.array([int(m["negative"].sum()) for m in masks])
    vol = np.array([float(torch.diff(torch.log(b.mid)).std()) for b in books])
    return {"invalid_steps": ladder + crossed + negative, "crossed": crossed, "volatility": vol}


def pick_cases(stats: dict, nearest: np.ndarray) -> list[tuple[str, int]]:
    """Five synthetic windows chosen by fixed rules, so the autopsy is not cherry-picked."""
    best_match = nearest.max(axis=0)          # each synthetic window's best SSIM to any real window
    order = np.argsort(best_match)
    cases = [("closest to real", int(order[-1])), ("typical (median SSIM)", int(order[len(order) // 2]))]
    if stats["invalid_steps"].max() > 0:
        cases.append(("most invalid steps", int(np.argmax(stats["invalid_steps"]))))
    else:
        cases.append(("least realistic depth", int(order[0])))
    cases.append(("flattest price path", int(np.argmin(stats["volatility"]))))
    cases.append(("most volatile", int(np.argmax(stats["volatility"]))))
    return cases


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

@dataclass
class RunAudit:
    """One run's numbers (summary.json) and the inputs of its figures."""

    summary: dict
    acf: dict[str, np.ndarray]       # squared-return ACF for real_train, real_test and synthetic
    real_books: list[OrderBook]      # every real test window
    fake_books: list[OrderBook]      # every synthetic window
    nearest: np.ndarray              # SSIM (n_real, n_fake), -1 outside the SSIM pool
    cases: list[tuple[str, int]]     # autopsy cases as (label, synthetic index)
    data_range: float                # log-volume range shared by every heatmap


def audit_run(model: SequenceGAN, splits: LOBSplits, run_name: str, checkpoint_step: int,
              n_samples: int, ssim_pool: int, seed: int, device: str) -> RunAudit:
    """Sample n_samples windows and measure them against the test hour.

    Every random choice is seeded with seed, so a rerun gives the same numbers.
    """
    test_x, test_anchor, real_books = decode_real_windows(splits, splits.test)
    _, _, train_books = decode_real_windows(splits, splits.train, splits.train.disjoint_indices())

    torch.manual_seed(seed)
    fake_x = model.sample(n_samples, splits.test.seq_len, device).cpu()
    fake_books = decode_synthetic_windows(splits, fake_x, test_anchor, torch.Generator().manual_seed(seed))

    # Nearest-match SSIM compares every real test window with every candidate,
    # so it runs on random pools of synthetic and training windows.
    data_range = depth_data_range(splits)
    rng = np.random.default_rng(seed)
    pool = rng.choice(len(fake_books), size=min(ssim_pool, len(fake_books)), replace=False)
    ref_pool = rng.choice(len(train_books), size=min(ssim_pool, len(train_books)), replace=False)
    real_imgs = [depth_image(b) for b in real_books]
    fake_imgs = [np.clip(depth_image(fake_books[i]), 0, data_range) for i in pool]
    ref_imgs = [depth_image(train_books[i]) for i in ref_pool]
    ssim_report, nearest_in_pool = ssim_scores(real_imgs, fake_imgs, ref_imgs, data_range, rng)

    pool_cases = pick_cases(window_stats([fake_books[i] for i in pool]), nearest_in_pool)
    cases = [(label, int(pool[j])) for label, j in pool_cases]
    nearest = np.full((len(real_books), len(fake_books)), -1.0)
    nearest[:, pool] = nearest_in_pool

    # Clustering is strong in the training hours and weak in the test hour, so both are references.
    books = {"real_train": train_books, "real_test": real_books, "synthetic": fake_books}
    acf = {name: pooled_acf(log_returns(b) ** 2) for name, b in books.items()}
    train_pool = splits.train.stack(range(0, len(splits.train), 4))[0]   # every 4th window keeps cdist small
    nearest_train = {"synthetic": float(nearest_train_distance(fake_x[:512], train_pool).median()),
                     "real_test": float(nearest_train_distance(test_x, train_pool).median())}

    summary = {
        "run": run_name,
        "model": type(model).__name__,
        "representation": splits.representation,
        "checkpoint_step": checkpoint_step,
        "parameters": model.parameter_counts()["total"],
        "violations": book_violations(fake_books),
        "kl_vs_test": kl_only(distribution_report(real_books, fake_books)),
        "kl_real_reference": kl_only(distribution_report(real_books, train_books)),
        "ssim": ssim_report,
        ACF_KEY: {k: [float(v[i]) for i in (0, 4, 9, 19)] for k, v in acf.items()},
        "acf_returns_lags_1_3": {name: pooled_acf(log_returns(b))[:3].tolist() for name, b in books.items()},
        "price_path": {name: price_path_stats(b) for name, b in books.items()},
        "diversity_rms": {"synthetic": mean_pairwise_distance(fake_x), "real_test": mean_pairwise_distance(test_x)},
        "nearest_train_rms_median": nearest_train,
        "autopsy_cases": [{"case": label, "synthetic_index": j, "best_ssim": float(nearest[:, j].max())}
                          for label, j in cases],
    }
    return RunAudit(summary, acf, real_books, fake_books, nearest, cases, data_range)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_example_book(book: OrderBook, levels: int = 3) -> None:
    """The first step of one decoded book."""
    print("Example synthetic book, first step (price $, size shares):")
    for k in range(levels):
        bid = f"{float(book.bid_price[0, k]):8.2f} x {float(book.bid_size[0, k]):6.0f}"
        ask = f"{float(book.ask_price[0, k]):8.2f} x {float(book.ask_size[0, k]):6.0f}"
        print(f"  level {k + 1}: bid {bid}   ask {ask}")


def print_summary(s: dict) -> None:
    """One run's summary, each number next to its real reference or target."""
    v, kl, ref, ss = s["violations"], s["kl_vs_test"], s["kl_real_reference"], s["ssim"]
    acf = {k: np.round(x, 3).tolist() for k, x in s[ACF_KEY].items()}
    print(f"  invalid books      crossed {v['crossed_rate']:.3f}  ladder {v['ladder_rate']:.3f}  "
          f"negative size {v['negative_size_rate']:.3f}")
    print(f"  KL vs test hour    spread {kl['kl_spread']:.3f}  return {kl['kl_return']:.3f}   "
          f"(real training vs test: {ref['kl_spread']:.3f}, {ref['kl_return']:.3f}; target <= {KL_TARGET})")
    print(f"  SSIM mean surface  {ss['mean_surface']:.3f}  "
          f"(real reference {ss['mean_surface_real_reference']:.3f}; target > {SSIM_TARGET})")
    print(f"  SSIM random pairs  {ss['random_pairs']:.3f}  (real reference {ss['random_pairs_real_reference']:.3f})")
    print(f"  SSIM nearest       {ss['nearest']:.3f}  (real reference {ss['nearest_real_reference']:.3f})")
    print(f"  ACF of r^2 lag 1/5/10/20   real train {acf['real_train']}   real test {acf['real_test']}   "
          f"synthetic {acf['synthetic']}")
    for name, label in [("real_train", "real train"), ("real_test", "real test"), ("synthetic", "synthetic")]:
        pp = s["price_path"][name]
        print(f"  price path {label:10s}  zero-move share {pp['zero_move_share']:.3f}   excursion from first mid "
              f"median {pp['excursion_ticks_median']:.1f} ticks, 90th percentile {pp['excursion_ticks_p90']:.1f} ticks")
    div, near = s["diversity_rms"], s["nearest_train_rms_median"]
    print(f"  diversity (RMS)   synthetic {div['synthetic']:.4f}  real {div['real_test']:.4f}")
    print(f"  nearest training window (median RMS)  synthetic {near['synthetic']:.4f}  "
          f"real test {near['real_test']:.4f}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

COMPARISON_COLUMNS = ["Run", "Params", "Crossed", "Ladder", "Spread KL", "Return KL", "SSIM mean surface",
                      "SSIM random", "SSIM nearest", "ACF r² lag 1", "ACF r² lag 10", "Zero-move share",
                      "Excursion median (ticks)", "Excursion p90 (ticks)"]


def _md_row(cells: list[str]) -> str:
    return "|" + "|".join(f" {c} " if c else " " for c in cells) + "|"


def _path_cells(pp: dict) -> list[str]:
    return [f"{pp['zero_move_share']:.3f}", f"{pp['excursion_ticks_median']:.1f}", f"{pp['excursion_ticks_p90']:.1f}"]


def _acf_cells(acf: list[float]) -> list[str]:
    return [f"{acf[0]:.3f}", f"{acf[2]:.3f}"]   # lags 1 and 10


def _run_row(s: dict) -> str:
    v, kl, ss = s["violations"], s["kl_vs_test"], s["ssim"]
    return _md_row([s["run"], f"{s['parameters']:,}", f"{v['crossed_rate']:.3f}", f"{v['ladder_rate']:.3f}",
                    f"{kl['kl_spread']:.3f}", f"{kl['kl_return']:.3f}", f"{ss['mean_surface']:.3f}",
                    f"{ss['random_pairs']:.3f}", f"{ss['nearest']:.3f}",
                    *_acf_cells(s[ACF_KEY]["synthetic"]), *_path_cells(s["price_path"]["synthetic"])])


def comparison_table(summaries: list[dict]) -> str:
    """Markdown table over several runs, with two real-data reference rows at the bottom.

    The reference rows come from the first run's summary.
    """
    ref = summaries[0]
    kl, ss = ref["kl_real_reference"], ref["ssim"]
    real_train = _md_row(["real training vs test (reference)", "", "0", "0", f"{kl['kl_spread']:.3f}",
                          f"{kl['kl_return']:.3f}", f"{ss['mean_surface_real_reference']:.3f}",
                          f"{ss['random_pairs_real_reference']:.3f}", f"{ss['nearest_real_reference']:.3f}",
                          *_acf_cells(ref[ACF_KEY]["real_train"]), *_path_cells(ref["price_path"]["real_train"])])
    real_test = _md_row(["real test hour (ACF and price path only)", *[""] * 8,
                         *_acf_cells(ref[ACF_KEY]["real_test"]), *_path_cells(ref["price_path"]["real_test"])])
    lines = [_md_row(COMPARISON_COLUMNS), "|" + "---|" * len(COMPARISON_COLUMNS),
             *(_run_row(s) for s in summaries), real_train, real_test]
    return "\n".join(lines) + "\n"
