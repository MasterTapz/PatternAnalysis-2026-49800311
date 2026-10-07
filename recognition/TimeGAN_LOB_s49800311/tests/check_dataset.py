"""Checks of src/dataset.py on the real LOBSTER session.

Every encoding must decode real books back exactly, the three periods must
be in time order, and a scaled real test window must decode to a valid book.

Run from the folder holding src/ and tests/:
  python tests/check_dataset.py --data-dir ../../../../data/LOBSTER
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import (REPRESENTATIONS, build_datasets, decode_book, encode_book, load_lobster,  # noqa: E402
                     subsample_events, trim_session)


def round_trip_check(data_dir: str, seq_len: int) -> None:
    """Encoding then decoding real books must give them back exactly, in every encoding."""
    book = subsample_events(trim_session(load_lobster(data_dir)), 10)
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


def split_check(data_dir: str, seq_len: int, representation: str, move_flag: bool) -> None:
    """The splits are in time order, and a scaled real window decodes to a valid book."""
    splits = build_datasets(data_dir, seq_len=seq_len, representation=representation, move_flag=move_flag)
    print(f"\n[{representation}, move_flag={move_flag}] {len(splits.feature_names)} features, config {splits.config}")

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
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args(argv)

    print("Exact round trip of every encoding on the real session:")
    round_trip_check(args.data_dir, args.seq_len)
    for representation in REPRESENTATIONS:
        for move_flag in (False, True):
            split_check(args.data_dir, args.seq_len, representation, move_flag)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
