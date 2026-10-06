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


# ---------------------------------------------------------------------------
# Feature encoding
#
# Two encodings of a book snapshot are supported, because the open research
# question is whether TimeGAN learns the order book constraints by itself:
#
#   "structured"  [mid log-return, log spread ticks, log ask gaps (L-1),
#                  log bid gaps (L-1), log1p ask sizes (L), log1p bid sizes (L)]
#                 A positive spread and monotonic ladder are built into the
#                 parametrisation; after tick rounding a spread or gap can
#                 still collapse to zero, which the audit reports as a locked
#                 book or a repeated level.
#   "raw"         [mid log-return, ask offsets from mid in ticks (L),
#                  bid offsets from mid in ticks (L), log1p sizes (2L)]
#                 Nothing prevents crossed books or broken ladders, so any
#                 validity the model shows here was learned.
#
# Row j of the features describes book row j + 1; book row 0 only anchors the
# first return, so its mid-price is passed back in when decoding.
# ---------------------------------------------------------------------------

REPRESENTATIONS = ("structured", "raw")


def feature_names(representation: str = "structured", levels: int = 10) -> list[str]:
    """Human-readable name of every feature column, in order."""
    if representation == "structured":
        prices = ["log_spread_ticks"]
        prices += [f"log_ask_gap_{k}_{k + 1}" for k in range(1, levels)]
        prices += [f"log_bid_gap_{k}_{k + 1}" for k in range(1, levels)]
    elif representation == "raw":
        prices = [f"ask_offset_{k}" for k in range(1, levels + 1)]
        prices += [f"bid_offset_{k}" for k in range(1, levels + 1)]
    else:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    sizes = [f"log1p_ask_size_{k}" for k in range(1, levels + 1)]
    sizes += [f"log1p_bid_size_{k}" for k in range(1, levels + 1)]
    return ["mid_log_return"] + prices + sizes


def encode_book(book: OrderBook, representation: str = "structured") -> torch.Tensor:
    """Encode book rows 1..N-1 as float64 features of shape (N - 1, F)."""
    mid = book.mid
    log_return = torch.log(mid[1:]) - torch.log(mid[:-1])
    ask, bid = book.ask_price[1:], book.bid_price[1:]

    if representation == "structured":
        # Real prices sit exactly on the tick grid, so rounding only removes
        # float noise from the division.
        spread_ticks = torch.round((ask[:, 0] - bid[:, 0]) / TICK)
        ask_gaps = torch.round((ask[:, 1:] - ask[:, :-1]) / TICK)
        bid_gaps = torch.round((bid[:, :-1] - bid[:, 1:]) / TICK)
        prices = torch.cat([torch.log(spread_ticks)[:, None], torch.log(ask_gaps), torch.log(bid_gaps)], dim=1)
    elif representation == "raw":
        centre = mid[1:, None]
        prices = torch.cat([(ask - centre) / TICK, (centre - bid) / TICK], dim=1)
    else:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")

    sizes = torch.log1p(torch.cat([book.ask_size[1:], book.bid_size[1:]], dim=1))
    return torch.cat([log_return[:, None], prices, sizes], dim=1)


def _snap(price: torch.Tensor) -> torch.Tensor:
    """Round dollar prices to the nearest tick."""
    return torch.round(price / TICK) * TICK


def decode_book(
    features: torch.Tensor,
    prev_mid: torch.Tensor | float,
    representation: str = "structured",
    levels: int = 10,
    round_to_tick: bool = True,
) -> OrderBook:
    """Invert encode_book for one sequence of shape (T, F).

    prev_mid is the mid-price just before the first row; the mid path is
    rebuilt by compounding the log-returns from it. No constraint is enforced
    here: whatever the features imply, including crossed books, negative
    sizes or repeated levels, is passed through for the audit to count.
    """
    f = features.double()
    if f.dim() != 2 or f.shape[1] != len(feature_names(representation, levels)):
        raise ValueError("features must have shape (T, F) for the given representation")
    mid = torch.as_tensor(prev_mid, dtype=torch.float64) * torch.exp(torch.cumsum(f[:, 0], dim=0))

    if representation == "structured":
        spread_ticks = torch.exp(f[:, 1])
        ask_gaps = torch.exp(f[:, 2:levels + 1])
        bid_gaps = torch.exp(f[:, levels + 1:2 * levels])
        if round_to_tick:
            spread_ticks, ask_gaps, bid_gaps = (torch.round(x) for x in (spread_ticks, ask_gaps, bid_gaps))
        best_bid = mid - spread_ticks * TICK / 2
        if round_to_tick:
            best_bid = _snap(best_bid)
        best_ask = best_bid + spread_ticks * TICK
        zero = torch.zeros_like(mid)[:, None]
        ask_price = best_ask[:, None] + torch.cat([zero, torch.cumsum(ask_gaps, dim=1)], dim=1) * TICK
        bid_price = best_bid[:, None] - torch.cat([zero, torch.cumsum(bid_gaps, dim=1)], dim=1) * TICK
        size_start = 2 * levels
    elif representation == "raw":
        ask_price = mid[:, None] + f[:, 1:levels + 1] * TICK
        bid_price = mid[:, None] - f[:, levels + 1:2 * levels + 1] * TICK
        if round_to_tick:
            ask_price, bid_price = _snap(ask_price), _snap(bid_price)
        size_start = 2 * levels + 1
    else:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")

    sizes = torch.expm1(f[:, size_start:])
    return OrderBook(
        time=torch.arange(f.shape[0], dtype=torch.float64),
        ask_price=ask_price,
        ask_size=sizes[:, :levels],
        bid_price=bid_price,
        bid_size=sizes[:, levels:],
    )
