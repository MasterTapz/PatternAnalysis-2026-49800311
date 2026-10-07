"""Every figure the project draws: training curves, heatmap autopsies and volatility clustering."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # write image files only; no display is needed, e.g. on a cluster node
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import TICK, OrderBook
from helpers.audit import depth_image
from helpers.metrics import KL_TARGET, step_violations


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def plot_history(history: dict, path: Path, kl_target: float = KL_TARGET) -> None:
    """Save one figure with the curves of every phase.

    Each phase keeps its own step counter, so each panel has its own x-axis.
    Validation KL panels show the spec target of 0.1 as a dashed line.
    """
    def series(kind: str, phase: str, key: str):
        rows = [r for r in history[kind] if r["phase"] == phase and key in r]
        return [r["step"] for r in rows], [r[key] for r in rows]

    panels = [
        ("Phase 1: reconstruction MSE", "autoencoder", [("train", "recon_mse", "train"), ("val", "val_recon_mse", "validation")], True),
        ("Phase 2: supervised MSE", "supervisor", [("train", "supervised_mse", "train"), ("val", "val_supervised_mse", "validation")], True),
        ("Phase 3: generator terms", "joint", [("train", "g_adv", "adversarial"), ("train", "g_adv_e", "adversarial, raw path"),
                                               ("train", "g_moment", "moment"), ("train", "g_supervised_mse", "supervised MSE")], True),
        ("Phase 3: discriminator", "joint", [("train", "d_real", "real BCE"), ("train", "d_fake", "fake BCE"),
                                             ("train", "d_updated", "share of steps updated")], False),
        ("Phase 3: gradient norms", "joint", [("train", "g_grad_norm", "generator"), ("train", "d_grad_norm", "discriminator")], True),
        ("Phase 3: autoencoder on validation", "joint", [("train", "e_recon_mse", "train"), ("val", "val_recon_mse", "validation")], True),
        ("Phase 3: KL to validation hour", "joint", [("val", "val_kl_spread", "spread"), ("val", "val_kl_return", "return")], True),
        ("Phase 3: invalid books", "joint", [("val", "crossed_rate", "crossed or locked"), ("val", "ladder_rate", "ladder"),
                                             ("val", "negative_size_rate", "negative size")], False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, (title, phase, lines, log_y) in zip(axes.flat, panels):
        drawn, plotted_values = False, []
        for kind, key, label in lines:
            x, y = series(kind, phase, key)
            if x:
                ax.plot(x, y, marker="o" if kind == "val" else None, markersize=3, label=label)
                plotted_values += y
                drawn = True
        if "KL" in title:
            ax.axhline(kl_target, color="grey", linestyle="--", linewidth=1, label=f"target {kl_target}")
        if drawn and log_y and all(v > 0 for v in plotted_values):   # log axis only when every value is positive
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
    """One row per case: real depth, synthetic depth, difference, and best quotes over time.

    nearest is the (n_real, n_fake) SSIM matrix; each synthetic window is shown
    next to the real test window it matches best.
    """
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
        _plot_best_quotes(axes[row, 3], real_books[r], fake_books[j])
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.975))   # leave room for the title above the first row
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_best_quotes(ax: plt.Axes, real: OrderBook, fake: OrderBook) -> None:
    """Best bid and ask of a real and a synthetic window in ticks from each one's first mid, invalid steps marked."""
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
    """ACF of squared returns for one run's synthetic windows against both real periods."""
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
    """ACF of squared returns for several runs (synthetic, keyed by run name) on one set of axes."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lags = np.arange(1, len(real_train) + 1)
    ax.plot(lags, real_train, "ko-", linewidth=2, label="real training hours")
    ax.plot(lags, real_test, "ko--", linewidth=1, label="real test hour")
    for run_name, curve in synthetic.items():
        ax.plot(lags, curve, label=run_name)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="lag (steps of 10 events)", ylabel="autocorrelation of squared returns", title="Volatility clustering by model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
