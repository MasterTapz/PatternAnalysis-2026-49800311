"""Microstructure validity and distribution metrics for real and generated books.

Shared by train.py (to track progress on the validation hour) and predict.py
(for the final audit). Every metric works on decoded books in dollars and
shares, so real and synthetic windows go through exactly the same code path.

Histogram units follow the market's own grid instead of arbitrary bins:
  * spreads are counted in whole ticks (1 tick = 0.01 dollars), with one extra
    bin at 0 that collects locked and crossed books (best bid >= best ask);
  * mid-price moves are counted in half-ticks, the smallest step the mid can
    take, so the return histogram has one bin per possible move size.
Log-returns at this scale are proportional to these moves (a half-tick move
is a log-return of about 2.3e-5 at 220 dollars), so the binning is a
resolution choice for the returns distribution, not a different quantity.
"""
from __future__ import annotations

import torch

from dataset import TICK, FeatureScaler, OrderBook, decode_book


def decode_windows(windows: torch.Tensor, scaler: FeatureScaler, anchors: torch.Tensor,
                   representation: str) -> list[OrderBook]:
    """Unscale and decode a batch of windows (B, T, F) into B order books.

    anchors[i] is the mid-price just before window i. For real windows it is
    their true previous mid; generated windows borrow anchors from real data,
    which only shifts the price level and leaves spreads and returns alone.
    """
    features = scaler.inverse_transform(windows.detach().cpu())
    anchors = torch.as_tensor(anchors, dtype=torch.float64)
    return [decode_book(features[i], anchors[i], representation) for i in range(len(features))]


def book_violations(books: list[OrderBook]) -> dict[str, float]:
    """Share of time steps that break each order book invariant.

    crossed      best bid >= best ask (locked or crossed, an arbitrage)
    ladder       ask prices not strictly rising or bid prices not strictly
                 falling across the 10 levels
    negative     any quoted size below zero
    any          at least one of the above
    """
    crossed = torch.cat([b.bid_price[:, 0] >= b.ask_price[:, 0] for b in books])
    ladder = torch.cat([(b.ask_price.diff(dim=1) <= 0).any(dim=1) | (b.bid_price.diff(dim=1) >= 0).any(dim=1)
                        for b in books])
    negative = torch.cat([(b.ask_size < 0).any(dim=1) | (b.bid_size < 0).any(dim=1) for b in books])
    return {
        "crossed_rate": crossed.double().mean().item(),
        "ladder_rate": ladder.double().mean().item(),
        "negative_size_rate": negative.double().mean().item(),
        "any_violation_rate": (crossed | ladder | negative).double().mean().item(),
    }


def spread_ticks(books: list[OrderBook]) -> torch.Tensor:
    """Best ask minus best bid in whole ticks, every time step of every book."""
    return torch.cat([torch.round(b.spread / TICK) for b in books]).long()


def mid_moves(books: list[OrderBook]) -> torch.Tensor:
    """Step-to-step mid-price changes in half-ticks, within each book only."""
    return torch.cat([torch.round(b.mid.diff() / (TICK / 2)) for b in books]).long()


def histogram_kl(p_values: torch.Tensor, q_values: torch.Tensor, low: int, high: int,
                 eps: float = 1e-6) -> float:
    """KL(P || Q) between two integer samples on the bins low..high.

    Values outside the range are clipped into the end bins, so tail mass is
    kept rather than dropped. eps is added to every bin before normalising,
    which keeps the divergence finite when Q has an empty bin that P uses.
    """
    def distribution(values: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(values.clamp(low, high) - low, minlength=high - low + 1).double()
        counts = counts + eps * counts.sum().clamp_min(1)
        return counts / counts.sum()

    p, q = distribution(p_values), distribution(q_values)
    return float((p * (p / q).log()).sum())


def distribution_report(real: list[OrderBook], fake: list[OrderBook]) -> dict[str, float]:
    """KL divergences of spreads and mid-price moves, in both directions.

    KL(real || synthetic) is the headline number for the spec target of 0.1;
    it penalises synthetic data that misses real behaviour. KL(synthetic ||
    real) penalises synthetic mass where real data has none, such as crossed
    books, so both are reported. Bin ranges come from the real data: spreads
    from 0 (locked or crossed) to the 99.5th percentile, moves symmetric to
    the 99.5th percentile of their absolute size.
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
