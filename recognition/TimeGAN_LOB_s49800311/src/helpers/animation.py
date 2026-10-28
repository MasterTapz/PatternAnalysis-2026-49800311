"""Order-book GIFs: real and synthetic windows animated step by step.

Each frame shows the ladder at one step, bids and asks as bars at their
distance from the current mid in ticks, and below it the mid and both side
VWAPs drawn up to that step, in ticks from the window's first mid. Axes are
fixed for the whole GIF so the motion is not hidden by rescaling.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # files only, no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from dataset import TICK, OrderBook
from helpers.metrics import side_vwap

STRIDE = 4      # steps per frame: 504 steps give 126 frames
FPS = 10
BID, ASK = "#2a78d6", "#d64541"


class BookFrames:
    """One window as arrays in ticks, ready to draw frame by frame."""

    def __init__(self, book: OrderBook):
        mid = book.mid
        self.bid_offset = ((book.bid_price - mid[:, None]) / TICK).numpy()
        self.ask_offset = ((book.ask_price - mid[:, None]) / TICK).numpy()
        self.bid_size = book.bid_size.clamp_min(0).numpy()
        self.ask_size = book.ask_size.clamp_min(0).numpy()
        start = float(mid[0])
        self.path = {"mid": ((mid - start) / TICK).numpy(),
                     "bid VWAP": ((side_vwap(book.bid_price, book.bid_size) - start) / TICK).numpy(),
                     "ask VWAP": ((side_vwap(book.ask_price, book.ask_size) - start) / TICK).numpy()}
        self.steps, self.levels = self.bid_offset.shape

    def reach(self) -> float:
        """Ticks from the mid that hold 99% of the level prices."""
        return float(np.percentile(np.abs(np.concatenate([self.bid_offset, self.ask_offset])), 99))

    def size_cap(self) -> float:
        return float(np.percentile(np.concatenate([self.bid_size, self.ask_size]), 99))

    def path_range(self) -> tuple[float, float]:
        values = np.concatenate(list(self.path.values()))
        return float(values.min()), float(values.max())


class Ladder:
    """Bars of one book on one axis, moved and resized every frame."""

    def __init__(self, ax: plt.Axes, frames: BookFrames, reach: float, size_cap: float, title: str):
        self.ax, self.frames, self.size_cap = ax, frames, size_cap
        levels = frames.levels
        self.bids = ax.bar(np.zeros(levels), np.zeros(levels), width=0.8, color=BID, label="bid")
        self.asks = ax.bar(np.zeros(levels), np.zeros(levels), width=0.8, color=ASK, label="ask")
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set(xlim=(-reach - 1, reach + 1), ylim=(0, size_cap * 1.05), title=title,
               xlabel="ticks from the mid", ylabel="shares")
        ax.legend(loc="upper right", fontsize=8)
        self.clock = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=9)

    def draw(self, t: int) -> None:
        f = self.frames
        for bars, offset, size in [(self.bids, f.bid_offset[t], f.bid_size[t]),
                                   (self.asks, f.ask_offset[t], f.ask_size[t])]:
            for bar, x, h in zip(bars, offset, size):
                bar.set_x(x - 0.4)
                bar.set_height(min(h, self.size_cap * 1.05))   # clip the rare giant queue
        self.clock.set_text(f"step {t}")


class PricePath:
    """Mid and side VWAPs of one or more books, drawn up to the current step."""

    STYLES = {"mid": ("black", 1.6), "bid VWAP": (BID, 1.2), "ask VWAP": (ASK, 1.2)}

    def __init__(self, ax: plt.Axes, books: list[tuple[str, BookFrames]], low: float, high: float, title: str):
        self.lines = []
        steps = max(f.steps for _, f in books)
        for (name, frames), dash in zip(books, ["-", "--"]):
            for key, (colour, width) in self.STYLES.items():
                label = f"{name} {key}" if len(books) > 1 else key
                (line,) = ax.plot([], [], dash, color=colour, linewidth=width, label=label)
                self.lines.append((line, frames.path[key]))
        pad = 0.05 * (high - low) + 1
        ax.set(xlim=(0, steps - 1), ylim=(low - pad, high + pad), title=title,
               xlabel="step", ylabel="ticks from the first mid")
        ax.legend(loc="upper left", fontsize=7, ncol=2)

    def draw(self, t: int) -> None:
        for line, series in self.lines:
            line.set_data(np.arange(t + 1), series[: t + 1])


def _shared_limits(frames: list[BookFrames]) -> tuple[float, float, float, float]:
    """Ladder reach and size cap, path low and high, shared by every panel of one stock."""
    low = min(f.path_range()[0] for f in frames)
    high = max(f.path_range()[1] for f in frames)
    return max(f.reach() for f in frames), max(f.size_cap() for f in frames), low, high


def _save(fig: plt.Figure, artists: list, steps: int, path: Path, stride: int, fps: int, dpi: int) -> None:
    def update(t: int) -> None:
        for artist in artists:
            artist.draw(t)

    animation = FuncAnimation(fig, update, frames=range(0, steps, stride))
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def animate_pair(real: OrderBook, fake: OrderBook, title: str, path: Path,
                 stride: int = STRIDE, fps: int = FPS) -> None:
    """Real window (left) and synthetic window (right), ladder on top and paths below.

    Both columns share their axes, so a synthetic path that drifts or a
    ladder that is too wide shows against the real one.
    """
    real_f, fake_f = BookFrames(real), BookFrames(fake)
    reach, cap, low, high = _shared_limits([real_f, fake_f])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), height_ratios=[1.1, 1])
    artists = [Ladder(axes[0, 0], real_f, reach, cap, "real test window"),
               Ladder(axes[0, 1], fake_f, reach, cap, "synthetic window"),
               PricePath(axes[1, 0], [("real", real_f)], low, high, "real mid and VWAPs"),
               PricePath(axes[1, 1], [("synthetic", fake_f)], low, high, "synthetic mid and VWAPs")]
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, artists, min(real_f.steps, fake_f.steps), path, stride, fps, dpi=80)


def animate_grid(pairs: dict[str, tuple[OrderBook, OrderBook]], title: str, path: Path,
                 stride: int = STRIDE, fps: int = FPS) -> None:
    """One column per stock: real ladder, synthetic ladder, then both paths together.

    Each stock keeps its own axes, since the five price scales differ by
    two orders of magnitude.
    """
    fig, axes = plt.subplots(3, len(pairs), figsize=(4.2 * len(pairs), 9.5), squeeze=False)
    artists, steps = [], []
    for col, (name, (real, fake)) in enumerate(pairs.items()):
        real_f, fake_f = BookFrames(real), BookFrames(fake)
        reach, cap, low, high = _shared_limits([real_f, fake_f])
        artists += [Ladder(axes[0, col], real_f, reach, cap, f"{name}: real"),
                    Ladder(axes[1, col], fake_f, reach, cap, f"{name}: synthetic"),
                    PricePath(axes[2, col], [("real", real_f), ("synthetic", fake_f)], low, high,
                              f"{name}: mid and VWAPs")]
        steps.append(min(real_f.steps, fake_f.steps))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, artists, min(steps), path, stride, fps, dpi=60)
