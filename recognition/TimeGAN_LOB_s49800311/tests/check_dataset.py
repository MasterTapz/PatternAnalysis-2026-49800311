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


def round_trip_check(data_dir: str, ticker: str, seq_len: int, clock_seconds: float) -> None:
    """Encoding then decoding real books must give them back exactly, in every encoding."""
    book = sampled_book(data_dir, ticker, clock_seconds)
    target = book.select(slice(1, None))
    for representation in REPRESENTATIONS:
        for move_flag in (False, True):
            x = encode_book(book, representation, move_flag)
            worst = 0.0
            for s in range(0, len(x), seq_len):
                d = decode_book(x[s:s + seq_len], book.mid[s], representation, move_flag=move_flag)
                t = target.select(slice(s, s + seq_len))
                worst = max(worst,
                            (d.ask_price - t.ask_price).abs().max().item(),
                            (d.bid_price - t.bid_price).abs().max().item(),
                            (d.mid - t.mid).abs().max().item())
            assert worst < 1e-9, f"{representation} move_flag={move_flag}: round trip off by {worst}"
            print(f"  round trip {representation:10s} move_flag={move_flag!s:5s}: "
                  f"max price error {worst:.1e} dollars")


def split_check(data_dir: str, ticker: str, seq_len: int, clock_seconds: float, representation: str,
                move_flag: bool) -> None:
    """The splits are in time order, and a scaled real window decodes to a valid book."""
    splits = build_datasets(data_dir, ticker=ticker, seq_len=seq_len, clock_seconds=clock_seconds,
                            representation=representation, move_flag=move_flag)
    print(f"\n[{ticker}, {representation}, move_flag={move_flag}] {len(splits.feature_names)} features, "
          f"config {splits.config}")

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
    decoded = decode_book(splits.scaler.inverse_transform(ds[0]), ds.anchor(0), representation, move_flag=move_flag)
    assert bool((decoded.ask_price[:, 0] > decoded.bid_price[:, 0]).all()), "real window decoded crossed"
    assert bool((decoded.ask_price.diff(dim=1) > 0).all() and (decoded.bid_price.diff(dim=1) < 0).all())
    print(f"  first test window decodes to a valid book, best bid/ask "
          f"{decoded.bid_price[0, 0]:.2f}/{decoded.ask_price[0, 0]:.2f}")


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
            for move_flag in (False, True):
                split_check(args.data_dir, ticker, args.seq_len, args.clock_seconds, representation, move_flag)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
