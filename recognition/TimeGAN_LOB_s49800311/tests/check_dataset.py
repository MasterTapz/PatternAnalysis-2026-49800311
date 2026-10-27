"""Checks of src/dataset.py on the real LOBSTER session.

Every encoding must decode real books back exactly, the three periods must
be in time order, and a scaled real test window must decode to a valid book.

Run from the folder holding src/ and tests/:
  python tests/check_dataset.py --data-dir ../../../../data/LOBSTER
  python tests/check_dataset.py --data-dir ../../../../data/LOBSTER --tickers AAPL AMZN GOOG INTC MSFT
  python tests/check_dataset.py --data-dir ../../../../data/LOBSTER --clock-seconds 1 --seq-len 504
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dataset import (REPRESENTATIONS, TICKERS, build_datasets, decode_book, encode_book,  # noqa: E402
                     load_lobster, sample_clock, session_bounds, subsample_events, trim_session)


def sampled_book(data_dir: str, ticker: str, clock_seconds: float):
    """The book as the pipeline samples it: every 10th event, or on the clock."""
    book = load_lobster(data_dir, ticker)
    if clock_seconds > 0:
        return sample_clock(book, clock_seconds, *session_bounds())
    return subsample_events(trim_session(book), 10)


def clock_check(data_dir: str, ticker: str, clock_seconds: float) -> None:
    """Each clock row is the book after the last event at or before its time."""
    events = load_lobster(data_dir, ticker)
    clocked = sample_clock(events, clock_seconds, *session_bounds())
    picks = torch.randperm(len(clocked), generator=torch.Generator().manual_seed(0))[:2000]
    for i in picks.tolist():
        t = float(clocked.time[i])
        row = int((events.time <= t).nonzero()[-1])
        assert torch.equal(clocked.ask_price[i], events.ask_price[row])
        assert torch.equal(clocked.bid_size[i], events.bid_size[row])
    steps = clocked.time.diff()
    assert bool((steps - clock_seconds).abs().max() < 1e-9)
    unchanged = float((clocked.mid.diff() == 0).double().mean())
    print(f"  clock {clock_seconds:g} s: {len(clocked)} rows from {clocked.time[0] / 3600:.3f}h, "
          f"2000 random rows match the last event before them, mid unchanged on {unchanged:.1%} of steps")


FLAG_SETTINGS = [(False, False), (True, False), (True, True)]   # (move_flag, hold_flags)


def round_trip_check(data_dir: str, ticker: str, seq_len: int, clock_seconds: float) -> None:
    """Encoding then decoding real books must give them back exactly, in every encoding."""
    book = sampled_book(data_dir, ticker, clock_seconds)
    target = book.select(slice(1, None))
    for representation in REPRESENTATIONS:
        for move_flag, hold_flags in FLAG_SETTINGS:
            x = encode_book(book, representation, move_flag, hold_flags)
            worst, worst_size = 0.0, 0.0
            for s in range(0, len(x), seq_len):
                d = decode_book(x[s:s + seq_len], book.mid[s], representation, move_flag=move_flag,
                                hold_flags=hold_flags)
                t = target.select(slice(s, s + seq_len))
                worst = max(worst,
                            (d.ask_price - t.ask_price).abs().max().item(),
                            (d.bid_price - t.bid_price).abs().max().item(),
                            (d.mid - t.mid).abs().max().item())
                worst_size = max(worst_size, (d.ask_size - t.ask_size).abs().max().item(),
                                 (d.bid_size - t.bid_size).abs().max().item())
            label = f"{representation} move_flag={move_flag} hold_flags={hold_flags}"
            assert worst < 1e-9, f"{label}: round trip off by {worst} dollars"
            assert worst_size < 1e-6, f"{label}: round trip off by {worst_size} shares"
            print(f"  round trip {representation:10s} move_flag={move_flag!s:5s} hold_flags={hold_flags!s:5s}: "
                  f"max price error {worst:.1e} dollars, size error {worst_size:.1e} shares")
    unchanged = encode_book(book, "structured", True, True)[:, 2:4].mean(dim=0)
    print(f"  real side-steps with no change at all: bid {unchanged[0]:.3f}, ask {unchanged[1]:.3f}")


def hold_check(splits) -> None:
    """Any hold flags at all, here random ones, must still decode a scaled real window to a valid book."""
    x = splits.scaler.inverse_transform(splits.test[0]).clone()
    columns = [splits.feature_names.index("bid_unchanged"), splits.feature_names.index("ask_unchanged")]
    x[:, columns] = torch.rand(len(x), 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    d = decode_book(x, splits.test.anchor(0), splits.representation, move_flag=splits.move_flag, hold_flags=True)
    held = float((x[:, columns] >= 0.5).double().mean())
    assert bool((d.ask_price[:, 0] > d.bid_price[:, 0]).all()), "held sides decoded crossed"
    assert bool((d.ask_price.diff(dim=1) > 0).all() and (d.bid_price.diff(dim=1) < 0).all())
    print(f"  random hold flags ({held:.0%} of side-steps held) still decode to a valid book")


def split_check(data_dir: str, ticker: str, seq_len: int, clock_seconds: float, representation: str,
                move_flag: bool, hold_flags: bool) -> None:
    """The splits are in time order, and a scaled real window decodes to a valid book."""
    splits = build_datasets(data_dir, ticker=ticker, seq_len=seq_len, clock_seconds=clock_seconds,
                            representation=representation, move_flag=move_flag, hold_flags=hold_flags)
    print(f"\n[{ticker}, {representation}, move_flag={move_flag}, hold_flags={hold_flags}] "
          f"{len(splits.feature_names)} features, config {splits.config}")

    for name in ("train", "val", "test"):
        ds = getattr(splits, name)
        t = ds.time
        outside = int(((ds.features < 0) | (ds.features > 1)).sum())
        print(f"  {name:5s} rows {len(ds.features):6d}  windows {len(ds):6d}  "
              f"{t[0] / 3600:6.3f}h to {t[-1] / 3600:6.3f}h  "
              f"values outside train range: {outside} of {ds.features.numel()}")
    assert splits.train.time[-1] < splits.val.time[0] < splits.val.time[-1] < splits.test.time[0]
    assert splits.train[0].shape == (seq_len, len(splits.feature_names))

    ds = splits.test
    decoded = decode_book(splits.scaler.inverse_transform(ds[0]), ds.anchor(0), representation, move_flag=move_flag,
                          hold_flags=hold_flags)
    assert bool((decoded.ask_price[:, 0] > decoded.bid_price[:, 0]).all()), "real window decoded crossed"
    assert bool((decoded.ask_price.diff(dim=1) > 0).all() and (decoded.bid_price.diff(dim=1) < 0).all())
    print(f"  first test window decodes to a valid book, best bid/ask "
          f"{decoded.bid_price[0, 0]:.2f}/{decoded.ask_price[0, 0]:.2f}")
    if hold_flags and representation == "structured":
        hold_check(splits)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the LOBSTER datasets and run sanity checks.")
    parser.add_argument("--data-dir", required=True, help="Folder holding the LOBSTER message and orderbook CSVs")
    parser.add_argument("--tickers", nargs="+", choices=TICKERS, default=["AMZN"])
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--clock-seconds", type=float, default=0, help="0 = every 10th event")
    args = parser.parse_args(argv)

    for ticker in args.tickers:
        print(f"\n=== {ticker} ===")
        if args.clock_seconds > 0:
            clock_check(args.data_dir, ticker, args.clock_seconds)
        print("Exact round trip of every encoding on the real session:")
        round_trip_check(args.data_dir, ticker, args.seq_len, args.clock_seconds)
        for representation in REPRESENTATIONS:
            for move_flag, hold_flags in FLAG_SETTINGS:
                split_check(args.data_dir, ticker, args.seq_len, args.clock_seconds, representation, move_flag,
                            hold_flags)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
