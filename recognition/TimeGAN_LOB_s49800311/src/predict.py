"""Load trained generators, sample synthetic order books and audit them against the test hour.

For every run given with --run this script:
  1. rebuilds the exact data pipeline stored in the checkpoint and loads best.pt
  2. generates synthetic windows and decodes them into order books (prices in
     dollars, sizes in shares), printing one example book
  3. scores them against the held-out test hour (15:00 to 15:50):
       invalid books     crossed or locked, broken ladder, negative size
       KL divergence     spreads (ticks) and mid moves (half-ticks), target <= 0.1
       SSIM              depth heatmaps (20 levels x time, log volume), target > 0.6
       volatility        autocorrelation of squared returns, lags 1 to 20
       price path        share of steps where the mid does not move, and how far
                         the mid wanders from its starting value within a window
       diversity         spread of samples and distance to the nearest training window
     with the same statistics for real training windows as a reference
  4. saves an autopsy figure of five rule-selected synthetic windows next to
     their closest real window, and an autocorrelation plot

With several runs, it also writes a comparison table and a combined
autocorrelation plot. NumPy and scikit-image are used here for SSIM and plots,
which the spec allows in predict.py.

Example (from the project folder, the one holding src/):
  python src/predict.py --data-dir ../../../../data/LOBSTER --run runs/structured_timegan_s0
  python src/predict.py --data-dir ../../../../data/LOBSTER --run runs/structured_timegan_s0 runs/structured_rgan_s0
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset import TICK, LOBSplits, OrderBook, build_datasets
from metrics import book_violations, decode_windows, distribution_report, mid_moves
from train import model_from_state

KL_TARGET, SSIM_TARGET = 0.1, 0.6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path, data_dir: str, device: str):
    """The best model of a run plus the data splits it was trained with."""
    path = run_dir / "best.pt" if (run_dir / "best.pt").exists() else run_dir / "last.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    wanted = inspect.signature(build_datasets).parameters
    config = {k: v for k, v in state["data_config"].items() if k in wanted}
    config["data_dir"] = data_dir
    splits = build_datasets(**config)
    return model_from_state(state, device), splits, state


def real_windows(ds, idx) -> tuple[torch.Tensor, torch.Tensor]:
    idx = list(idx)
    return torch.stack([ds[i] for i in idx]), torch.tensor([ds.anchor(i) for i in idx], dtype=torch.float64)


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


def ssim(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """SSIM (Wang et al., 2004) with a 7x7 window and a fixed data range for every pair."""
    return float(structural_similarity(a, b, data_range=data_range, win_size=7))


def ssim_scores(real_imgs: list, fake_imgs: list, ref_imgs: list, data_range: float, rng: np.random.Generator) -> dict:
    """Three protocols, each with a real-data reference (real training windows vs the test hour).

    random        mean SSIM over random (real test, synthetic) pairs
    nearest       for every real test window, the best SSIM over the synthetic pool
    mean_surface  SSIM between the average synthetic and average real depth surface

    Generated windows are unconditional, so no synthetic window is meant to
    match a particular real one. On this data even two real windows reach only
    about 0.1 under random pairing, so the 0.6 target is only attainable for
    the average surface; all three are reported so the choice is visible.
    """
    n_pairs = 1024
    i = rng.integers(len(real_imgs), size=n_pairs)
    random_fake = np.mean([ssim(real_imgs[a], fake_imgs[b], data_range) for a, b in zip(i, rng.integers(len(fake_imgs), size=n_pairs))])
    random_ref = np.mean([ssim(real_imgs[a], ref_imgs[b], data_range) for a, b in zip(i, rng.integers(len(ref_imgs), size=n_pairs))])
    nearest = np.array([[ssim(r, f, data_range) for f in fake_imgs] for r in real_imgs])      # (n_real, n_fake)
    nearest_ref = np.array([[ssim(r, f, data_range) for f in ref_imgs] for r in real_imgs])
    real_mean = np.mean(real_imgs, axis=0)
    return {
        "random_pairs": float(random_fake), "random_pairs_real_reference": float(random_ref),
        "nearest": float(nearest.max(axis=1).mean()), "nearest_real_reference": float(nearest_ref.max(axis=1).mean()),
        "mean_surface": ssim(np.mean(fake_imgs, axis=0), real_mean, data_range),
        "mean_surface_real_reference": ssim(np.mean(ref_imgs, axis=0), real_mean, data_range),
        "_matrix": nearest,
    }


# ---------------------------------------------------------------------------
# Volatility clustering, diversity and memorisation
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
# One run
# ---------------------------------------------------------------------------

def window_stats(books: list[OrderBook]) -> dict[str, np.ndarray]:
    """Per-window counts used to pick the autopsy cases."""
    ladder = np.array([int(((b.ask_price.diff(dim=1) <= 0).any(1) | (b.bid_price.diff(dim=1) >= 0).any(1)).sum()) for b in books])
    crossed = np.array([int((b.bid_price[:, 0] >= b.ask_price[:, 0]).sum()) for b in books])
    negative = np.array([int(((b.ask_size < 0).any(1) | (b.bid_size < 0).any(1)).sum()) for b in books])
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


def plot_autopsy(cases, fake_books, real_books, nearest, data_range: float, title: str, path: Path) -> None:
    """One row per case: real depth, synthetic depth, difference, and best quotes over time."""
    fig, axes = plt.subplots(len(cases), 4, figsize=(19, 3.1 * len(cases)), squeeze=False)
    for row, (label, j) in enumerate(cases):
        r = int(np.argmax(nearest[:, j]))
        real_img, fake_img = depth_image(real_books[r]), depth_image(fake_books[j])
        kwargs = dict(aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=data_range)
        axes[row, 0].imshow(real_img, **kwargs)
        axes[row, 0].set_title(f"{label}: closest real test window")
        im = axes[row, 1].imshow(fake_img, **kwargs)
        axes[row, 1].set_title(f"synthetic, SSIM {nearest[r, j]:.2f}")
        diff = axes[row, 2].imshow(fake_img - real_img, aspect="auto", origin="lower", cmap="RdBu_r",
                                   vmin=-data_range / 2, vmax=data_range / 2)
        axes[row, 2].set_title("synthetic minus real (log volume)")
        for ax in axes[row, :3]:
            ax.set_yticks([0, 9, 10, 19], ["bid 10", "bid 1", "ask 1", "ask 10"])
            ax.axhline(9.5, color="white", linewidth=0.8)
        fig.colorbar(im, ax=axes[row, 1], fraction=0.04)
        fig.colorbar(diff, ax=axes[row, 2], fraction=0.04)

        ax = axes[row, 3]
        for book, style, name in [(real_books[r], "--", "real"), (fake_books[j], "-", "synthetic")]:
            ref = float(book.mid[0])
            ax.plot((book.ask_price[:, 0] - ref) / TICK, style, color="tab:red", label=f"{name} best ask")
            ax.plot((book.bid_price[:, 0] - ref) / TICK, style, color="tab:blue", label=f"{name} best bid")
        fb = fake_books[j]
        bad = ((fb.ask_price.diff(dim=1) <= 0).any(1) | (fb.bid_price.diff(dim=1) >= 0).any(1) |
               (fb.bid_price[:, 0] >= fb.ask_price[:, 0])).nonzero().flatten()
        if len(bad):
            ax.scatter(bad, torch.zeros(len(bad)), marker="|", color="black", s=60, label="invalid step")
        ax.set_title("best quotes, ticks from first mid")
        ax.set_xlabel("time step")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.975))   # leave room for the title above the first row
    fig.savefig(path, dpi=110)
    plt.close(fig)


def audit_run(run_dir: Path, args) -> dict:
    model, splits, state = load_run(run_dir, args.data_dir, args.device)
    rep = state["data_config"]["representation"]
    out_dir = run_dir / "predict"
    out_dir.mkdir(exist_ok=True)
    print(f"\n=== {run_dir.name}: {type(model).__name__}, {rep} encoding, checkpoint step {state['step']} ===")

    test_x, test_anchor = real_windows(splits.test, range(len(splits.test)))
    train_x, train_anchor = real_windows(splits.train, range(0, len(splits.train), splits.train.seq_len))
    move_flag = splits.config.get("move_flag", False)
    real_books = decode_windows(test_x, splits.scaler, test_anchor, rep, move_flag)
    train_books = decode_windows(train_x, splits.scaler, train_anchor, rep, move_flag)

    torch.manual_seed(args.seed)
    fake_x = model.sample(args.n_samples, splits.test.seq_len, args.device).cpu()
    anchors = test_anchor[torch.randint(len(test_anchor), (args.n_samples,), generator=torch.Generator().manual_seed(args.seed))]
    fake_books = decode_windows(fake_x, splits.scaler, anchors, rep, move_flag)

    b = fake_books[0]
    print("Example synthetic book, first step (price $, size shares):")
    for k in range(3):
        print(f"  level {k + 1}: bid {float(b.bid_price[0, k]):8.2f} x {float(b.bid_size[0, k]):6.0f}   "
              f"ask {float(b.ask_price[0, k]):8.2f} x {float(b.ask_size[0, k]):6.0f}")

    # SSIM on depth heatmaps, with a fixed range: the largest log volume seen in training.
    size_cols = [i for i, n in enumerate(splits.feature_names) if "size" in n]
    data_range = float((splits.scaler.shift + splits.scaler.scale)[size_cols].max())
    rng = np.random.default_rng(args.seed)
    pool = rng.choice(len(fake_books), size=min(args.ssim_pool, len(fake_books)), replace=False)
    ref_pool = rng.choice(len(train_books), size=min(args.ssim_pool, len(train_books)), replace=False)
    real_imgs = [depth_image(x) for x in real_books]
    fake_imgs = [np.clip(depth_image(fake_books[i]), 0, data_range) for i in pool]
    ref_imgs = [depth_image(train_books[i]) for i in ref_pool]
    ssim_report = ssim_scores(real_imgs, fake_imgs, ref_imgs, data_range, rng)
    nearest = ssim_report.pop("_matrix")

    # Volatility clustering differs by period on this day (strong in the training
    # hours, weak in the test hour), so synthetic data is compared with both.
    acf = {"real_train": pooled_acf(log_returns(train_books) ** 2), "real_test": pooled_acf(log_returns(real_books) ** 2),
           "synthetic": pooled_acf(log_returns(fake_books) ** 2)}
    acf_returns = {k: pooled_acf(log_returns(b))[:3].tolist()
                   for k, b in [("real_train", train_books), ("real_test", real_books), ("synthetic", fake_books)]}
    train_all = torch.stack([splits.train[i] for i in range(0, len(splits.train), 4)])
    summary = {
        "run": run_dir.name, "model": type(model).__name__, "representation": rep, "checkpoint_step": state["step"],
        "parameters": model.parameter_counts()["total"],
        "violations": book_violations(fake_books),
        "kl_vs_test": {k: v for k, v in distribution_report(real_books, fake_books).items() if k.startswith("kl")},
        "kl_real_reference": {k: v for k, v in distribution_report(real_books, train_books).items() if k.startswith("kl")},
        "ssim": ssim_report,
        "acf_squared_returns_lags_1_5_10_20": {k: [float(v[i]) for i in (0, 4, 9, 19)] for k, v in acf.items()},
        "acf_returns_lags_1_3": acf_returns,
        "price_path": {k: price_path_stats(b)
                       for k, b in [("real_train", train_books), ("real_test", real_books), ("synthetic", fake_books)]},
        "diversity_rms": {"synthetic": mean_pairwise_distance(fake_x), "real_test": mean_pairwise_distance(test_x)},
        "nearest_train_rms_median": {"synthetic": float(nearest_train_distance(fake_x[:512], train_all).median()),
                                     "real_test": float(nearest_train_distance(test_x, train_all).median())},
    }

    stats = window_stats([fake_books[i] for i in pool])
    cases = [(label, int(pool[j])) for label, j in pick_cases(stats, nearest)]
    nearest_full = np.full((len(real_books), len(fake_books)), -1.0)
    nearest_full[:, pool] = nearest
    plot_autopsy(cases, fake_books, real_books, nearest_full, data_range,
                 f"{run_dir.name}: five rule-selected synthetic windows vs their closest real test window",
                 out_dir / "autopsy.png")
    summary["autopsy_cases"] = [{"case": label, "synthetic_index": j, "best_ssim": float(nearest_full[:, j].max())}
                                for label, j in cases]

    fig, ax = plt.subplots(figsize=(7, 4))
    lags = np.arange(1, 21)
    ax.plot(lags, acf["real_train"], "o-", label="real training hours")
    ax.plot(lags, acf["real_test"], "o--", label="real test hour")
    ax.plot(lags, acf["synthetic"], "s-", label="synthetic")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="lag (steps of 10 events)", ylabel="autocorrelation of squared returns",
           title=f"Volatility clustering: {run_dir.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "volatility_acf.png", dpi=120)
    plt.close(fig)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print_summary(summary)
    summary["_acf"] = acf
    return summary


def print_summary(s: dict) -> None:
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


def write_comparison(summaries: list[dict], out_dir: Path) -> None:
    """Markdown table and combined autocorrelation plot over several runs.

    The price path columns (zero-move share, median and 90th percentile
    excursion in ticks) describe each run's synthetic windows; the two
    reference rows give the same numbers for real training and real test windows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
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
    # Written as UTF-8 so the superscript in the header survives on Windows.
    (out_dir / "comparison.md").write_text(header + "\n".join(rows) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    lags = np.arange(1, 21)
    ax.plot(lags, ref["_acf"]["real_train"], "ko-", linewidth=2, label="real training hours")
    ax.plot(lags, ref["_acf"]["real_test"], "ko--", linewidth=1, label="real test hour")
    for s in summaries:
        ax.plot(lags, s["_acf"]["synthetic"], label=s["run"])
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="lag (steps of 10 events)", ylabel="autocorrelation of squared returns", title="Volatility clustering by model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "volatility_acf_comparison.png", dpi=120)
    plt.close(fig)
    print(f"\nComparison written to {out_dir / 'comparison.md'}\n")
    print((out_dir / "comparison.md").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Folder holding the LOBSTER CSVs")
    p.add_argument("--run", nargs="+", required=True, help="One or more run folders produced by train.py")
    p.add_argument("--n-samples", type=int, default=1024)
    p.add_argument("--ssim-pool", type=int, default=256, help="Synthetic and training windows used for nearest-match SSIM")
    p.add_argument("--out-dir", default="runs/comparison", help="Where the multi-run comparison goes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    summaries = [audit_run(Path(r), args) for r in args.run]
    if len(summaries) > 1:
        write_comparison(summaries, Path(args.out_dir))


if __name__ == "__main__":
    main()
