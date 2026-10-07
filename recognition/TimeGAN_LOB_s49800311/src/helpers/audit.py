"""The test-hour audit behind predict.py: what one run's synthetic books get right and wrong.

audit_run samples a trained model, decodes the windows into order books and
measures them against the real test hour, with real training windows as the
reference for what is achievable:
    invalid books     crossed or locked, broken ladder, negative size
    KL divergence     spreads (ticks) and mid moves (half-ticks), target <= 0.1
    SSIM              depth heatmaps (20 levels x time, log volume), target > 0.6
    volatility        autocorrelation of squared returns, lags 1 to 20
    price path        share of steps where the mid does not move, and how far
                      the mid wanders from its starting value within a window
    diversity         spread of samples and distance to the nearest training window
It also picks the autopsy cases by rule. NumPy and scikit-image are used here
for SSIM, which the spec allows on the prediction side.
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

SSIM_TARGET = 0.6   # spec target for SSIM between real and synthetic depth snapshots


# ---------------------------------------------------------------------------
# Depth heatmaps and SSIM
# ---------------------------------------------------------------------------

def depth_image(book: OrderBook) -> np.ndarray:
    """A (20, T) depth heatmap: rows bid 10..1 then ask 1..10, values log1p(size).

    Price rises up the image, so the spread sits between rows 10 and 11.
    Negative sizes are clipped to zero here; they are counted separately.
    """
    bid = torch.log1p(book.bid_size.clamp_min(0)).flip(1)    # level 10 first
    ask = torch.log1p(book.ask_size.clamp_min(0))
    return torch.cat([bid, ask], dim=1).T.numpy()


def depth_data_range(splits: LOBSplits) -> float:
    """One fixed SSIM data range for every heatmap: the largest log volume seen in training."""
    size_cols = [i for i, name in enumerate(splits.feature_names) if "size" in name]
    return float((splits.scaler.shift + splits.scaler.scale)[size_cols].max())


def ssim(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """SSIM (Wang et al., 2004) with a 7x7 window and a fixed data range for every pair."""
    return float(structural_similarity(a, b, data_range=data_range, win_size=7))


def ssim_scores(real_imgs: list[np.ndarray], fake_imgs: list[np.ndarray], ref_imgs: list[np.ndarray],
                data_range: float, rng: np.random.Generator) -> tuple[dict[str, float], np.ndarray]:
    """Three protocols, each with a real-data reference (real training windows vs the test hour).

    random        mean SSIM over random (real test, synthetic) pairs
    nearest       for every real test window, the best SSIM over the synthetic pool
    mean_surface  SSIM between the average synthetic and average real depth surface

    Generated windows are unconditional, so no synthetic window is meant to
    match a particular real one. On this data even two real windows reach only
    about 0.1 under random pairing, so the 0.6 target is only attainable for
    the average surface; all three are reported so the choice is visible.
    Returns the scores and the (n_real, n_fake) SSIM matrix used for nearest.
    """
    n_pairs = 1024
    i = rng.integers(len(real_imgs), size=n_pairs)
    random_fake = np.mean([ssim(real_imgs[a], fake_imgs[b], data_range) for a, b in zip(i, rng.integers(len(fake_imgs), size=n_pairs))])
    random_ref = np.mean([ssim(real_imgs[a], ref_imgs[b], data_range) for a, b in zip(i, rng.integers(len(ref_imgs), size=n_pairs))])
    nearest = np.array([[ssim(r, f, data_range) for f in fake_imgs] for r in real_imgs])      # (n_real, n_fake)
    nearest_ref = np.array([[ssim(r, f, data_range) for f in ref_imgs] for r in real_imgs])
    real_mean = np.mean(real_imgs, axis=0)
    scores = {
        "random_pairs": float(random_fake), "random_pairs_real_reference": float(random_ref),
        "nearest": float(nearest.max(axis=1).mean()), "nearest_real_reference": float(nearest_ref.max(axis=1).mean()),
        "mean_surface": ssim(np.mean(fake_imgs, axis=0), real_mean, data_range),
        "mean_surface_real_reference": ssim(np.mean(ref_imgs, axis=0), real_mean, data_range),
    }
    return scores, nearest


# ---------------------------------------------------------------------------
# Volatility clustering, price paths, diversity and memorisation
# ---------------------------------------------------------------------------

def log_returns(books: list[OrderBook]) -> torch.Tensor:
    """Within-window mid-price log-returns, shape (n_books, T - 1)."""
    return torch.stack([torch.diff(torch.log(b.mid)) for b in books])


def pooled_acf(x: torch.Tensor, max_lag: int = 20) -> np.ndarray:
    """Autocorrelation at lags 1..max_lag, pooled over windows (no pair crosses a window edge)."""
    x = x - x.mean()
    var = (x * x).mean()
    return np.array([float((x[:, :-k] * x[:, k:]).mean() / var) for k in range(1, max_lag + 1)])


def price_path_stats(books: list[OrderBook]) -> dict[str, float]:
    """How often the mid stands still and how far it wanders, per set of windows.

    zero_move_share     share of within-window steps whose mid move, rounded
                        to half-ticks, is zero (real best quotes move in steps
                        and hold flat; a generator that jitters every step
                        scores low here)
    excursion_ticks_*   per window, the largest absolute distance of the mid
                        from the window's first mid, in ticks; the median and
                        90th percentile over windows (a generator whose price
                        paths drift or trend scores high here)
    Both use the decoded books, so real and synthetic windows are measured the same way.
    """
    moves = mid_moves(books)
    excursion = torch.stack([(b.mid - b.mid[0]).abs().max() / TICK for b in books]).double()
    return {
        "zero_move_share": float((moves == 0).double().mean()),
        "excursion_ticks_median": float(torch.quantile(excursion, 0.5)),   # quantile interpolates; median() would not
        "excursion_ticks_p90": float(torch.quantile(excursion, 0.9)),
    }


def mean_pairwise_distance(x: torch.Tensor, n: int = 512, seed: int = 0) -> float:
    """Average RMS distance between windows (B, T, F); low values mean low diversity."""
    g = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(len(x), generator=g)[:n]].flatten(1).double()
    d = torch.cdist(x, x) / np.sqrt(x.shape[1])
    return float(d[~torch.eye(len(x), dtype=torch.bool)].mean())


def nearest_train_distance(x: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    """RMS distance from every window in x to its closest training window."""
    a, b = x.flatten(1).double(), train.flatten(1).double()
    return torch.cdist(a, b).min(dim=1).values / np.sqrt(a.shape[1])


# ---------------------------------------------------------------------------
# Autopsy case selection
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
    """Five synthetic windows chosen by rule, so the autopsy is not cherry-picked."""
    best_match = nearest.max(axis=0)                   # each synthetic window's best SSIM to any real window
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
    """One run's audit: the numbers for summary.json and the inputs of its figures."""

    summary: dict                    # JSON-ready, written to predict/summary.json
    acf: dict[str, np.ndarray]       # ACF of squared returns at lags 1..20 for real_train, real_test, synthetic
    real_books: list[OrderBook]      # every real test window
    fake_books: list[OrderBook]      # every synthetic window
    nearest: np.ndarray              # SSIM (n_real, n_fake); -1 where a synthetic window is outside the SSIM pool
    cases: list[tuple[str, int]]     # autopsy cases as (label, synthetic window index)
    data_range: float                # log-volume range shared by every depth heatmap


def audit_run(model: SequenceGAN, splits: LOBSplits, run_name: str, checkpoint_step: int,
              n_samples: int, ssim_pool: int, seed: int, device: str) -> RunAudit:
    """Sample n_samples synthetic windows and measure them against the test hour.

    Every random choice (sampling, borrowed anchors, SSIM pools and pairs) is
    seeded with seed, so re-running on the same checkpoint gives the same numbers.
    """
    test_x, test_anchor, real_books = decode_real_windows(splits, splits.test)
    _, _, train_books = decode_real_windows(splits, splits.train, splits.train.disjoint_indices())

    torch.manual_seed(seed)
    fake_x = model.sample(n_samples, splits.test.seq_len, device).cpu()
    fake_books = decode_synthetic_windows(splits, fake_x, test_anchor, torch.Generator().manual_seed(seed))

    # SSIM on depth heatmaps. Nearest-match SSIM compares every real test window with
    # every candidate, so it runs on a random pool of synthetic and of training windows.
    data_range = depth_data_range(splits)
    rng = np.random.default_rng(seed)
    pool = rng.choice(len(fake_books), size=min(ssim_pool, len(fake_books)), replace=False)
    ref_pool = rng.choice(len(train_books), size=min(ssim_pool, len(train_books)), replace=False)
    real_imgs = [depth_image(b) for b in real_books]
    fake_imgs = [np.clip(depth_image(fake_books[i]), 0, data_range) for i in pool]
    ref_imgs = [depth_image(train_books[i]) for i in ref_pool]
    ssim_report, nearest_in_pool = ssim_scores(real_imgs, fake_imgs, ref_imgs, data_range, rng)

    cases = [(label, int(pool[j])) for label, j in pick_cases(window_stats([fake_books[i] for i in pool]), nearest_in_pool)]
    nearest = np.full((len(real_books), len(fake_books)), -1.0)
    nearest[:, pool] = nearest_in_pool

    # Volatility clustering differs by period on this day (strong in the training
    # hours, weak in the test hour), so synthetic data is compared with both.
    books = {"real_train": train_books, "real_test": real_books, "synthetic": fake_books}
    acf = {name: pooled_acf(log_returns(b) ** 2) for name, b in books.items()}
    train_pool = splits.train.stack(range(0, len(splits.train), 4))[0]   # every 4th window keeps cdist affordable
    summary = {
        "run": run_name, "model": type(model).__name__, "representation": splits.representation,
        "checkpoint_step": checkpoint_step,
        "parameters": model.parameter_counts()["total"],
        "violations": book_violations(fake_books),
        "kl_vs_test": kl_only(distribution_report(real_books, fake_books)),
        "kl_real_reference": kl_only(distribution_report(real_books, train_books)),
        "ssim": ssim_report,
        "acf_squared_returns_lags_1_5_10_20": {k: [float(v[i]) for i in (0, 4, 9, 19)] for k, v in acf.items()},
        "acf_returns_lags_1_3": {name: pooled_acf(log_returns(b))[:3].tolist() for name, b in books.items()},
        "price_path": {name: price_path_stats(b) for name, b in books.items()},
        "diversity_rms": {"synthetic": mean_pairwise_distance(fake_x), "real_test": mean_pairwise_distance(test_x)},
        "nearest_train_rms_median": {"synthetic": float(nearest_train_distance(fake_x[:512], train_pool).median()),
                                     "real_test": float(nearest_train_distance(test_x, train_pool).median())},
        "autopsy_cases": [{"case": label, "synthetic_index": j, "best_ssim": float(nearest[:, j].max())}
                          for label, j in cases],
    }
    return RunAudit(summary, acf, real_books, fake_books, nearest, cases, data_range)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_example_book(book: OrderBook, levels: int = 3) -> None:
    """The first time step of one decoded book, best levels first."""
    print("Example synthetic book, first step (price $, size shares):")
    for k in range(levels):
        print(f"  level {k + 1}: bid {float(book.bid_price[0, k]):8.2f} x {float(book.bid_size[0, k]):6.0f}   "
              f"ask {float(book.ask_price[0, k]):8.2f} x {float(book.ask_size[0, k]):6.0f}")


def print_summary(s: dict) -> None:
    """Console view of one run's summary, each number next to its real reference or target."""
    v, kl, ref, ss = s["violations"], s["kl_vs_test"], s["kl_real_reference"], s["ssim"]
    a = s["acf_squared_returns_lags_1_5_10_20"]
    print(f"  invalid books      crossed {v['crossed_rate']:.3f}  ladder {v['ladder_rate']:.3f}  negative size {v['negative_size_rate']:.3f}")
    print(f"  KL vs test hour    spread {kl['kl_spread']:.3f}  return {kl['kl_return']:.3f}   "
          f"(real training vs test: {ref['kl_spread']:.3f}, {ref['kl_return']:.3f}; target <= {KL_TARGET})")
    print(f"  SSIM mean surface  {ss['mean_surface']:.3f}  (real reference {ss['mean_surface_real_reference']:.3f}; target > {SSIM_TARGET})")
    print(f"  SSIM random pairs  {ss['random_pairs']:.3f}  (real reference {ss['random_pairs_real_reference']:.3f})")
    print(f"  SSIM nearest       {ss['nearest']:.3f}  (real reference {ss['nearest_real_reference']:.3f})")
    print(f"  ACF of r^2 lag 1/5/10/20   real train {np.round(a['real_train'], 3).tolist()}   real test "
          f"{np.round(a['real_test'], 3).tolist()}   synthetic {np.round(a['synthetic'], 3).tolist()}")
    for name, label in [("real_train", "real train"), ("real_test", "real test"), ("synthetic", "synthetic")]:
        pp = s["price_path"][name]
        print(f"  price path {label:10s}  zero-move share {pp['zero_move_share']:.3f}   excursion from first mid "
              f"median {pp['excursion_ticks_median']:.1f} ticks, 90th percentile {pp['excursion_ticks_p90']:.1f} ticks")
    print(f"  diversity (RMS)   synthetic {s['diversity_rms']['synthetic']:.4f}  real {s['diversity_rms']['real_test']:.4f}")
    print(f"  nearest training window (median RMS)  synthetic {s['nearest_train_rms_median']['synthetic']:.4f}  "
          f"real test {s['nearest_train_rms_median']['real_test']:.4f}")


def comparison_table(summaries: list[dict]) -> str:
    """Markdown table over several runs, with real-data reference rows at the bottom.

    The price path columns (zero-move share, median and 90th percentile
    excursion in ticks) describe each run's synthetic windows; the two
    reference rows give the same numbers for real training and real test windows.
    """
    header = ("| Run | Params | Crossed | Ladder | Spread KL | Return KL | SSIM mean surface | SSIM random | SSIM nearest "
              "| ACF r² lag 1 | ACF r² lag 10 | Zero-move share | Excursion median (ticks) | Excursion p90 (ticks) |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    acf_key = "acf_squared_returns_lags_1_5_10_20"

    def path_cells(pp: dict) -> str:
        return f"{pp['zero_move_share']:.3f} | {pp['excursion_ticks_median']:.1f} | {pp['excursion_ticks_p90']:.1f}"

    rows = [f"| {s['run']} | {s['parameters']:,} | {s['violations']['crossed_rate']:.3f} | {s['violations']['ladder_rate']:.3f} | "
            f"{s['kl_vs_test']['kl_spread']:.3f} | {s['kl_vs_test']['kl_return']:.3f} | {s['ssim']['mean_surface']:.3f} | "
            f"{s['ssim']['random_pairs']:.3f} | {s['ssim']['nearest']:.3f} | "
            f"{s[acf_key]['synthetic'][0]:.3f} | {s[acf_key]['synthetic'][2]:.3f} | "
            f"{path_cells(s['price_path']['synthetic'])} |" for s in summaries]
    ref = summaries[0]
    rows.append(f"| real training vs test (reference) | | 0 | 0 | {ref['kl_real_reference']['kl_spread']:.3f} | "
                f"{ref['kl_real_reference']['kl_return']:.3f} | {ref['ssim']['mean_surface_real_reference']:.3f} | "
                f"{ref['ssim']['random_pairs_real_reference']:.3f} | {ref['ssim']['nearest_real_reference']:.3f} | "
                f"{ref[acf_key]['real_train'][0]:.3f} | {ref[acf_key]['real_train'][2]:.3f} | "
                f"{path_cells(ref['price_path']['real_train'])} |")
    rows.append(f"| real test hour (ACF and price path only) | | | | | | | | | {ref[acf_key]['real_test'][0]:.3f} | "
                f"{ref[acf_key]['real_test'][2]:.3f} | {path_cells(ref['price_path']['real_test'])} |")
    return header + "\n".join(rows) + "\n"
