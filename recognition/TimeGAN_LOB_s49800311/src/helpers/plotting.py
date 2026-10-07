"""Training curves, heatmap autopsies, volatility plots and the README figures."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # files only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import TICK, OrderBook
from helpers.audit import depth_image
from helpers.metrics import KL_TARGET, step_violations


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# (title, phase, [(history kind, key, label)], log y-axis)
HISTORY_PANELS = [
    ("Phase 1: reconstruction MSE", "autoencoder",
     [("train", "recon_mse", "train"), ("val", "val_recon_mse", "validation")], True),
    ("Phase 2: supervised MSE", "supervisor",
     [("train", "supervised_mse", "train"), ("val", "val_supervised_mse", "validation")], True),
    ("Phase 3: generator terms", "joint",
     [("train", "g_adv", "adversarial"), ("train", "g_adv_e", "adversarial, raw path"),
      ("train", "g_moment", "moment"), ("train", "g_supervised_mse", "supervised MSE")], True),
    ("Phase 3: discriminator", "joint",
     [("train", "d_real", "real BCE"), ("train", "d_fake", "fake BCE"),
      ("train", "d_updated", "share of steps updated")], False),
    ("Phase 3: gradient norms", "joint",
     [("train", "g_grad_norm", "generator"), ("train", "d_grad_norm", "discriminator")], True),
    ("Phase 3: autoencoder on validation", "joint",
     [("train", "e_recon_mse", "train"), ("val", "val_recon_mse", "validation")], True),
    ("Phase 3: KL to validation hour", "joint",
     [("val", "val_kl_spread", "spread"), ("val", "val_kl_return", "return")], True),
    ("Phase 3: invalid books", "joint",
     [("val", "crossed_rate", "crossed or locked"), ("val", "ladder_rate", "ladder"),
      ("val", "negative_size_rate", "negative size")], False),
]


def plot_history(history: dict, path: Path, kl_target: float = KL_TARGET) -> None:
    """One figure with the curves of every phase; each phase has its own step axis."""
    def series(kind: str, phase: str, key: str):
        rows = [r for r in history[kind] if r["phase"] == phase and key in r]
        return [r["step"] for r in rows], [r[key] for r in rows]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, (title, phase, lines, log_y) in zip(axes.flat, HISTORY_PANELS):
        values = []
        for kind, key, label in lines:
            x, y = series(kind, phase, key)
            if x:
                ax.plot(x, y, marker="o" if kind == "val" else None, markersize=3, label=label)
                values += y
        drawn = bool(values)
        if "KL" in title:
            ax.axhline(kl_target, color="grey", linestyle="--", linewidth=1, label=f"target {kl_target}")
        if drawn and log_y and all(v > 0 for v in values):
            ax.set_yscale("log")
        ax.set_title(title if drawn else f"{title} (not run)")
        ax.set_xlabel("step in phase")
        if drawn:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def plot_autopsy(cases: list[tuple[str, int]], fake_books: list[OrderBook], real_books: list[OrderBook],
                 nearest: np.ndarray, data_range: float, title: str, path: Path) -> None:
    """One row per case: real depth, synthetic depth, their difference and the best quotes.

    Each synthetic window is shown next to the real test window it matches
    best in nearest, the (n_real, n_fake) SSIM matrix.
    """
    fig, axes = plt.subplots(len(cases), 4, figsize=(19, 3.1 * len(cases)), squeeze=False)
    heatmap = dict(aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=data_range)
    for row, (label, j) in enumerate(cases):
        r = int(np.argmax(nearest[:, j]))
        real_img, fake_img = depth_image(real_books[r]), depth_image(fake_books[j])
        axes[row, 0].imshow(real_img, **heatmap)
        axes[row, 0].set_title(f"{label}: closest real test window")
        im = axes[row, 1].imshow(fake_img, **heatmap)
        axes[row, 1].set_title(f"synthetic, SSIM {nearest[r, j]:.2f}")
        diff = axes[row, 2].imshow(fake_img - real_img, aspect="auto", origin="lower", cmap="RdBu_r",
                                   vmin=-data_range / 2, vmax=data_range / 2)
        axes[row, 2].set_title("synthetic minus real (log volume)")
        for ax in axes[row, :3]:
            ax.set_yticks([0, 9, 10, 19], ["bid 10", "bid 1", "ask 1", "ask 10"])
            ax.axhline(9.5, color="white", linewidth=0.8)
        fig.colorbar(im, ax=axes[row, 1], fraction=0.04)
        fig.colorbar(diff, ax=axes[row, 2], fraction=0.04)
        _plot_best_quotes(axes[row, 3], real_books[r], fake_books[j])
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.975))   # room for the title
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_best_quotes(ax: plt.Axes, real: OrderBook, fake: OrderBook) -> None:
    """Best bid and ask in ticks from each window's first mid, with invalid synthetic steps marked."""
    for book, style, name in [(real, "--", "real"), (fake, "-", "synthetic")]:
        ref = float(book.mid[0])
        ax.plot((book.ask_price[:, 0] - ref) / TICK, style, color="tab:red", label=f"{name} best ask")
        ax.plot((book.bid_price[:, 0] - ref) / TICK, style, color="tab:blue", label=f"{name} best bid")
    masks = step_violations(fake)
    bad = (masks["ladder"] | masks["crossed"]).nonzero().flatten()
    if len(bad):
        ax.scatter(bad, torch.zeros(len(bad)), marker="|", color="black", s=60, label="invalid step")
    ax.set_title("best quotes, ticks from first mid")
    ax.set_xlabel("time step")
    ax.legend(fontsize=7, ncol=2)


def plot_volatility_acf(acf: dict[str, np.ndarray], run_name: str, path: Path) -> None:
    """Squared-return ACF of one run against both real periods."""
    fig, ax = plt.subplots(figsize=(7, 4))
    lags = np.arange(1, len(acf["synthetic"]) + 1)
    ax.plot(lags, acf["real_train"], "o-", label="real training hours")
    ax.plot(lags, acf["real_test"], "o--", label="real test hour")
    ax.plot(lags, acf["synthetic"], "s-", label="synthetic")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="lag (steps of 10 events)", ylabel="autocorrelation of squared returns",
           title=f"Volatility clustering: {run_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_volatility_comparison(real_train: np.ndarray, real_test: np.ndarray,
                               synthetic: dict[str, np.ndarray], path: Path) -> None:
    """Squared-return ACF of several runs (keyed by run name) on one plot."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lags = np.arange(1, len(real_train) + 1)
    ax.plot(lags, real_train, "ko-", linewidth=2, label="real training hours")
    ax.plot(lags, real_test, "ko--", linewidth=1, label="real test hour")
    for run_name, curve in synthetic.items():
        ax.plot(lags, curve, label=run_name)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="lag (steps of 10 events)", ylabel="autocorrelation of squared returns",
           title="Volatility clustering by model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# README figures
# ---------------------------------------------------------------------------

# README file name -> source inside the runs folder
README_COPIES = {
    "training_curves_structured_timegan.png": "structured_timegan_s0/training_curves.png",
    "autopsy_structured_timegan.png": "structured_timegan_s0/predict/autopsy.png",
    "autopsy_raw_timegan.png": "raw_timegan_s0/predict/autopsy.png",
    "autopsy_structured_rganm.png": "structured_rganm_s0/predict/autopsy.png",
    "volatility_acf_comparison.png": "comparison/volatility_acf_comparison.png",
}
LADDER_FIGURE = "ladder_rate_raw_joint_phase.png"

# (run folder, label, colour, marker, line style). Colour-blind safe colours,
# and markers and line styles differ too, so colour is never the only cue.
LADDER_RUNS = [
    ("raw_timegan_s0", "TimeGAN", "#2a78d6", "o", "-"),
    ("raw_nosup_s0", "TimeGAN without supervisor", "#eb6834", "s", "--"),
    ("raw_rganm_s0", "recurrent GAN + moment loss", "#1baf7a", "^", ":"),
]


def export_readme_assets(runs_dir: Path, out_dir: Path) -> list[Path]:
    """Copy the README figures out of the runs folder and draw the ladder figure.

    Only PNGs and history files are read, no model. Returns every PNG in out_dir.
    """
    out_dir.mkdir(exist_ok=True)
    for name, source in README_COPIES.items():
        shutil.copy2(runs_dir / source, out_dir / name)
    plot_ladder(runs_dir, out_dir / LADDER_FIGURE)
    return sorted(out_dir.glob("*.png"))


def joint_validation(history_path: Path) -> list[dict]:
    """Joint-phase validation records in step order."""
    history = json.loads(history_path.read_text())
    return sorted((r for r in history["val"] if r["phase"] == "joint"), key=lambda r: r["step"])


def best_step(records: list[dict]) -> int:
    """The step kept as best.pt: the last record flagged new_best."""
    return [r["step"] for r in records if r.get("new_best")][-1]


def plot_ladder(runs_dir: Path, path: Path) -> None:
    """Broken-ladder rate and validation spread KL over the joint phase, best.pt ringed."""
    fig, (ax_ladder, ax_kl) = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True)
    for run, label, colour, marker, style in LADDER_RUNS:
        records = joint_validation(runs_dir / run / "history.json")
        steps = [r["step"] for r in records]
        best = best_step(records)
        for ax, key in [(ax_ladder, "ladder_rate"), (ax_kl, "val_kl_spread")]:
            values = [r[key] for r in records]
            ax.plot(steps, values, linestyle=style, linewidth=2, color=colour, marker=marker, markersize=5,
                    label=label)
            ax.plot([best], [values[steps.index(best)]], marker="o", markersize=13, markerfacecolor="none",
                    markeredgecolor=colour, markeredgewidth=1.5, linestyle="none")

    ax_ladder.set(title="Broken ladder rate on generated books (raw encoding)", xlabel="joint-phase step",
                  ylabel="share of time steps with a broken ladder", ylim=(0, 1.05))
    ax_kl.axhline(KL_TARGET, color="#6b6b6b", linestyle="--", linewidth=1)
    ax_kl.text(4_300, KL_TARGET - 0.005, f"target {KL_TARGET}", color="#4a4a4a", fontsize=8, ha="left", va="top")
    # The baseline starts at 3.06, so the axis is cut to keep the other runs readable.
    ax_kl.set(title="Spread KL against the validation hour (axis cut at 0.45)", xlabel="joint-phase step",
              ylabel="KL(real || synthetic)", ylim=(0, 0.45))
    for ax in (ax_ladder, ax_kl):
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_ladder.legend(fontsize=8, loc="lower right", title="ring = best.pt", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
