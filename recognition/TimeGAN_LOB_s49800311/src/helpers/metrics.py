"""Validity and distribution metrics on decoded order books (PyTorch only).

Real and synthetic windows go through the same decoding and the same
metrics. Histograms use the market's own grid: spreads in whole ticks, with
bin 0 collecting locked and crossed books, and mid moves in half-ticks, the
smallest step the mid can take.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from dataset import HALF_TICK, TICK, FeatureScaler, LOBSplits, LOBWindowDataset, OrderBook, decode_book

KL_TARGET = 0.1   # spec target for KL(real || synthetic) of spreads and returns


# ---------------------------------------------------------------------------
# Scaled windows to order books
# ---------------------------------------------------------------------------

def decode_windows(windows: torch.Tensor, scaler: FeatureScaler, anchors: torch.Tensor,
                   representation: str, move_flag: bool = False) -> list[OrderBook]:
    """Unscale and decode windows (B, T, F) into B books.

    anchors[i] is the mid-price before window i. Synthetic windows borrow
    real anchors, which only shifts the price level.
    """
    features = scaler.inverse_transform(windows.detach().cpu())
    anchors = torch.as_tensor(anchors, dtype=torch.float64)
    return [decode_book(features[i], anchors[i], representation, move_flag=move_flag)
            for i in range(len(features))]


def decode_real_windows(splits: LOBSplits, dataset: LOBWindowDataset, indices: Iterable[int] | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, list[OrderBook]]:
    """Real windows (n, T, F), their anchors (n,) and the decoded books."""
    windows, anchors = dataset.stack(indices)
    books = decode_windows(windows, splits.scaler, anchors, splits.representation, splits.move_flag)
    return windows, anchors, books


def decode_synthetic_windows(splits: LOBSplits, windows: torch.Tensor, real_anchors: torch.Tensor,
                             generator: torch.Generator) -> list[OrderBook]:
    """Decode generated windows, each at a randomly drawn real anchor."""
    pick = torch.randint(len(real_anchors), (len(windows),), generator=generator)
    return decode_windows(windows, splits.scaler, real_anchors[pick], splits.representation, splits.move_flag)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def step_violations(book: OrderBook) -> dict[str, torch.Tensor]:
    """Boolean masks (T,) of the steps that break each rule.

    crossed   best bid >= best ask
    ladder    ask prices not strictly rising or bid prices not strictly falling
    negative  any size below zero
    """
    ask_ladder_broken = (book.ask_price.diff(dim=1) <= 0).any(dim=1)
    bid_ladder_broken = (book.bid_price.diff(dim=1) >= 0).any(dim=1)
    return {
        "crossed": book.bid_price[:, 0] >= book.ask_price[:, 0],
        "ladder": ask_ladder_broken | bid_ladder_broken,
        "negative": (book.ask_size < 0).any(dim=1) | (book.bid_size < 0).any(dim=1),
    }


def book_violations(books: list[OrderBook]) -> dict[str, float]:
    """Share of steps breaking each rule, and any rule."""
    masks = [step_violations(b) for b in books]
    crossed, ladder, negative = (torch.cat([m[kind] for m in masks]) for kind in ("crossed", "ladder", "negative"))
    return {
        "crossed_rate": crossed.double().mean().item(),
        "ladder_rate": ladder.double().mean().item(),
        "negative_size_rate": negative.double().mean().item(),
        "any_violation_rate": (crossed | ladder | negative).double().mean().item(),
    }


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def spread_ticks(books: list[OrderBook]) -> torch.Tensor:
    """Spread in whole ticks at every step of every book."""
    return torch.cat([torch.round(b.spread / TICK) for b in books]).long()


def mid_moves(books: list[OrderBook]) -> torch.Tensor:
    """Step-to-step mid changes in half-ticks, within each book."""
    return torch.cat([torch.round(b.mid.diff() / HALF_TICK) for b in books]).long()


def histogram_kl(p_values: torch.Tensor, q_values: torch.Tensor, low: int, high: int,
                 eps: float = 1e-6) -> float:
    """KL(P || Q) of two integer samples over the bins low..high.

    Values outside the range go into the end bins. eps keeps the result
    finite when Q has an empty bin that P uses.
    """
    def distribution(values: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(values.clamp(low, high) - low, minlength=high - low + 1).double()
        counts = counts + eps * counts.sum().clamp_min(1)
        return counts / counts.sum()

    p, q = distribution(p_values), distribution(q_values)
    return float((p * (p / q).log()).sum())


def distribution_report(real: list[OrderBook], fake: list[OrderBook]) -> dict[str, float]:
    """KL of spreads and mid moves in both directions.

    KL(real || synthetic) is the spec's number; the reverse direction also
    punishes synthetic mass where real data has none. Bin ranges run up to the
    real 99.5th percentile.
    """
    real_spread, fake_spread = spread_ticks(real), spread_ticks(fake)
    real_move, fake_move = mid_moves(real), mid_moves(fake)
    spread_high = max(1, int(torch.quantile(real_spread.double(), 0.995)))
    move_high = max(1, int(torch.quantile(real_move.abs().double(), 0.995)))
    return {
        "kl_spread": histogram_kl(real_spread, fake_spread, 0, spread_high),
        "kl_return": histogram_kl(real_move, fake_move, -move_high, move_high),
        "kl_spread_reverse": histogram_kl(fake_spread, real_spread, 0, spread_high),
        "kl_return_reverse": histogram_kl(fake_move, real_move, -move_high, move_high),
        "spread_bins": spread_high + 1,
        "return_bins": 2 * move_high + 1,
    }


def kl_only(report: dict[str, float]) -> dict[str, float]:
    """The four KL entries of a distribution_report."""
    return {k: v for k, v in report.items() if k.startswith("kl")}
