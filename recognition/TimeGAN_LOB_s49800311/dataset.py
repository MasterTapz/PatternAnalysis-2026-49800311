"""Data loading and preprocessing for the LOBSTER limit order book sample.

The LOBSTER sample (AMZN, 2012-06-21, 10 levels) ships as two header-less CSVs:
a message file with one row per order book event, and an order book file whose
row i is the state of the book right after message i. This module turns those
files into fixed-length feature windows for TimeGAN while keeping the training,
validation and test periods in strict chronological order.

Column layout of the order book file, repeated for levels 1 to 10:
    ask price, ask size, bid price, bid size
Prices are dollars times 10,000. Level 1 is the best quote on each side.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

PRICE_SCALE = 10_000             # LOBSTER stores dollars times 10,000
TICK = 0.01                      # AMZN minimum price increment in dollars
EMPTY_ASK = 9_999_999_999        # LOBSTER dummy price for an unoccupied ask level
EMPTY_BID = -9_999_999_999       # LOBSTER dummy price for an unoccupied bid level
HALT_EVENT = 7                   # message type marking a trading halt
MARKET_OPEN = 9.5 * 3600         # 09:30 in seconds after midnight
MARKET_CLOSE = 16 * 3600         # 16:00 in seconds after midnight


def clock_to_seconds(hhmm: str) -> float:
    """Convert a 'HH:MM' wall-clock string to seconds after midnight."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 3600 + int(minutes) * 60


@dataclass
class OrderBook:
    """A time-ordered sequence of order book snapshots.

    Prices are float64 dollars so tick arithmetic stays exact, sizes are shares.
    Every price and size tensor has shape (N, levels), with level 1 in column 0.
    """

    time: torch.Tensor        # (N,) seconds after midnight
    ask_price: torch.Tensor
    ask_size: torch.Tensor
    bid_price: torch.Tensor
    bid_size: torch.Tensor

    def __len__(self) -> int:
        return self.time.shape[0]

    @property
    def levels(self) -> int:
        return self.ask_price.shape[1]

    @property
    def mid(self) -> torch.Tensor:
        """Mid-price, the average of the best ask and best bid."""
        return (self.ask_price[:, 0] + self.bid_price[:, 0]) / 2

    @property
    def spread(self) -> torch.Tensor:
        """Best ask minus best bid in dollars. Positive in a valid book."""
        return self.ask_price[:, 0] - self.bid_price[:, 0]

    def select(self, index) -> "OrderBook":
        """Rows picked by a slice, boolean mask or index tensor, as contiguous copies."""
        return OrderBook(
            time=self.time[index].contiguous(),
            ask_price=self.ask_price[index].contiguous(),
            ask_size=self.ask_size[index].contiguous(),
            bid_price=self.bid_price[index].contiguous(),
            bid_size=self.bid_size[index].contiguous(),
        )


def _find_one(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {data_dir}, found {len(matches)}"
        )
    return matches[0]


def load_lobster(data_dir: str | Path, levels: int = 10) -> OrderBook:
    """Read a LOBSTER message/order book file pair into an OrderBook.

    Only the event timestamps are taken from the message file; the book file
    already holds the state after every event. Unoccupied levels and trading
    halts are rejected rather than silently patched, because both would break
    the price ladder that the rest of the pipeline relies on. The AMZN sample
    contains neither.
    """
    data_dir = Path(data_dir)
    message_path = _find_one(data_dir, f"*_message_{levels}.csv")
    book_path = _find_one(data_dir, f"*_orderbook_{levels}.csv")

    messages = pd.read_csv(message_path, header=None, usecols=[0, 1], names=["time", "type"])
    book = torch.tensor(pd.read_csv(book_path, header=None).to_numpy(), dtype=torch.int64)

    if len(messages) != book.shape[0]:
        raise ValueError("Message and order book files have different row counts")
    if book.shape[1] != 4 * levels:
        raise ValueError(f"Expected {4 * levels} order book columns, found {book.shape[1]}")
    if (messages["type"] == HALT_EVENT).any():
        raise ValueError("Trading halt found; split the session around it before loading")

    ask_raw, bid_raw = book[:, 0::4], book[:, 2::4]
    if (ask_raw == EMPTY_ASK).any() or (bid_raw == EMPTY_BID).any():
        raise ValueError("Unoccupied price levels found; request fewer levels or mask them")

    return OrderBook(
        time=torch.tensor(messages["time"].to_numpy(), dtype=torch.float64),
        ask_price=ask_raw.double() / PRICE_SCALE,
        ask_size=book[:, 1::4].double(),
        bid_price=bid_raw.double() / PRICE_SCALE,
        bid_size=book[:, 3::4].double(),
    )


def trim_session(book: OrderBook, open_minutes: float = 10, close_minutes: float = 10) -> OrderBook:
    """Drop the first and last minutes of continuous trading.

    On the AMZN sample the first 10 minutes run at about twice the usual spread
    and the last 10 minutes at about twice the usual event rate, so both behave
    unlike the rest of the day.
    """
    start = MARKET_OPEN + open_minutes * 60
    end = MARKET_CLOSE - close_minutes * 60
    return book.select((book.time >= start) & (book.time < end))


def subsample_events(book: OrderBook, stride: int) -> OrderBook:
    """Keep every stride-th snapshot, so one step spans a fixed number of events.

    Sampling by events rather than by clock time keeps the event-driven nature
    of the book: busy periods produce more steps per minute than quiet ones.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    return book.select(slice(None, None, stride))


def split_by_time(time: torch.Tensor, train_end: str = "14:00", val_end: str = "15:00") -> dict[str, slice]:
    """Contiguous train, validation and test row ranges cut at wall-clock times.

    Training uses the earliest hours and testing the latest, so no model ever
    sees data from after the period it is evaluated on.
    """
    if not bool((time[1:] >= time[:-1]).all()):
        raise ValueError("Timestamps must be sorted before splitting")
    t_train, t_val = clock_to_seconds(train_end), clock_to_seconds(val_end)
    if not t_train < t_val:
        raise ValueError("train_end must come before val_end")
    i_train = int(torch.searchsorted(time, torch.tensor(t_train, dtype=time.dtype)))
    i_val = int(torch.searchsorted(time, torch.tensor(t_val, dtype=time.dtype)))
    return {
        "train": slice(0, i_train),
        "val": slice(i_train, i_val),
        "test": slice(i_val, len(time)),
    }
