"""Loading and preprocessing of the LOBSTER AMZN level 10 sample.

LOBSTER ships two CSVs without headers: a message file (one row per event)
and an order book file whose row i is the book right after message i. Each
book row holds ask price, ask size, bid price and bid size for levels 1 to
10, with prices in dollars times 10,000.

The pipeline is load -> trim -> subsample -> encode -> split -> scale -> window,
and the train, validation and test periods stay in time order.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

PRICE_SCALE = 10_000             # LOBSTER prices are dollars times 10,000
TICK = 0.01                      # AMZN tick size in dollars
HALF_TICK = TICK / 2             # smallest possible mid-price move
EMPTY_ASK = 9_999_999_999        # LOBSTER placeholder for an empty ask level
EMPTY_BID = -9_999_999_999       # LOBSTER placeholder for an empty bid level
HALT_EVENT = 7                   # message type of a trading halt
MARKET_OPEN = 9.5 * 3600         # 09:30 in seconds after midnight
MARKET_CLOSE = 16 * 3600         # 16:00 in seconds after midnight
REPRESENTATIONS = ("structured", "raw")


def clock_to_seconds(hhmm: str) -> float:
    """'HH:MM' to seconds after midnight."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 3600 + int(minutes) * 60


@dataclass
class OrderBook:
    """A sequence of book snapshots, prices in dollars and sizes in shares.

    Price and size tensors have shape (N, levels) with level 1 in column 0.
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
        return (self.ask_price[:, 0] + self.bid_price[:, 0]) / 2

    @property
    def spread(self) -> torch.Tensor:
        return self.ask_price[:, 0] - self.bid_price[:, 0]

    def select(self, index: slice | torch.Tensor) -> OrderBook:
        """The rows picked by a slice, mask or index tensor."""
        return OrderBook(
            time=self.time[index].contiguous(),
            ask_price=self.ask_price[index].contiguous(),
            ask_size=self.ask_size[index].contiguous(),
            bid_price=self.bid_price[index].contiguous(),
            bid_size=self.bid_size[index].contiguous(),
        )


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def _find_one(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file matching {pattern!r} in {data_dir}, found {len(matches)}")
    return matches[0]


def load_lobster(data_dir: str | Path, levels: int = 10) -> OrderBook:
    """Read the message and order book files of one LOBSTER day.

    Only the timestamps come from the message file. Empty levels and trading
    halts raise an error instead of being patched; the AMZN sample has neither.
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
        raise ValueError("The session contains a trading halt")

    ask_raw, bid_raw = book[:, 0::4], book[:, 2::4]
    if (ask_raw == EMPTY_ASK).any() or (bid_raw == EMPTY_BID).any():
        raise ValueError("The book has empty price levels; load fewer levels")

    return OrderBook(
        time=torch.tensor(messages["time"].to_numpy(), dtype=torch.float64),
        ask_price=ask_raw.double() / PRICE_SCALE,
        ask_size=book[:, 1::4].double(),
        bid_price=bid_raw.double() / PRICE_SCALE,
        bid_size=book[:, 3::4].double(),
    )


def trim_session(book: OrderBook, open_minutes: float = 10, close_minutes: float = 10) -> OrderBook:
    """Drop the first and last minutes of trading.

    On this day the opening minutes have about twice the usual spread and the
    closing minutes about twice the usual event rate.
    """
    start = MARKET_OPEN + open_minutes * 60
    end = MARKET_CLOSE - close_minutes * 60
    return book.select((book.time >= start) & (book.time < end))


def subsample_events(book: OrderBook, stride: int) -> OrderBook:
    """Keep every stride-th snapshot, so one step is a fixed number of events."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    return book.select(slice(None, None, stride))


def split_by_time(time: torch.Tensor, train_end: str = "14:00", val_end: str = "15:00") -> dict[str, slice]:
    """Train, validation and test row ranges cut at clock times."""
    if not bool((time[1:] >= time[:-1]).all()):
        raise ValueError("Timestamps must be sorted")
    t_train, t_val = clock_to_seconds(train_end), clock_to_seconds(val_end)
    if t_train >= t_val:
        raise ValueError("train_end must come before val_end")
    i_train = int(torch.searchsorted(time, torch.tensor(t_train, dtype=time.dtype)))
    i_val = int(torch.searchsorted(time, torch.tensor(t_val, dtype=time.dtype)))
    return {"train": slice(0, i_train), "val": slice(i_train, i_val), "test": slice(i_val, len(time))}


# ---------------------------------------------------------------------------
# Feature encoding
#
# structured  [mid log-return, log spread in ticks, log ask gaps (L-1),
#              log bid gaps (L-1), log1p ask sizes (L), log1p bid sizes (L)]
#             The parametrisation keeps the spread positive and the ladder
#             ordered, up to tick rounding.
# raw         [mid log-return, ask offsets from mid in ticks (L),
#              bid offsets from mid in ticks (L), log1p sizes (2L)]
#             Nothing stops crossed books or broken ladders, so any validity
#             the model shows here is learned.
#
# Feature row j describes book row j + 1. Book row 0 only anchors the first
# return, so decoding needs the mid-price before the window (prev_mid).
#
# move_flag=True adds a binary "mid_moved" column after the log-return. About
# half of the real steps have no mid move, and a generator with continuous
# outputs almost never lands on exactly zero, so the flag lets the model
# decide "move or not" separately from the size of the move.
# ---------------------------------------------------------------------------

def feature_names(representation: str = "structured", levels: int = 10, move_flag: bool = False) -> list[str]:
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
    flag = ["mid_moved"] if move_flag else []
    return ["mid_log_return"] + flag + prices + sizes


def encode_book(book: OrderBook, representation: str = "structured", move_flag: bool = False) -> torch.Tensor:
    """Features of book rows 1..N-1, shape (N - 1, F), float64."""
    mid = book.mid
    columns = [(torch.log(mid[1:]) - torch.log(mid[:-1]))[:, None]]
    if move_flag:
        moved = torch.round(torch.diff(mid) / HALF_TICK) != 0
        columns.append(moved.double()[:, None])
    ask, bid = book.ask_price[1:], book.bid_price[1:]

    if representation == "structured":
        # Real prices are already on the tick grid; rounding only removes float error.
        spread = torch.round((ask[:, 0] - bid[:, 0]) / TICK)
        ask_gaps = torch.round((ask[:, 1:] - ask[:, :-1]) / TICK)
        bid_gaps = torch.round((bid[:, :-1] - bid[:, 1:]) / TICK)
        columns += [torch.log(spread)[:, None], torch.log(ask_gaps), torch.log(bid_gaps)]
    elif representation == "raw":
        centre = mid[1:, None]
        columns += [(ask - centre) / TICK, (centre - bid) / TICK]
    else:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")

    columns.append(torch.log1p(torch.cat([book.ask_size[1:], book.bid_size[1:]], dim=1)))
    return torch.cat(columns, dim=1)


def _snap(price: torch.Tensor) -> torch.Tensor:
    """Round dollar prices to the nearest tick."""
    return torch.round(price / TICK) * TICK


def _flagged_mid_path(log_returns: torch.Tensor, moved: torch.Tensor, prev_mid: float,
                      round_to_tick: bool) -> torch.Tensor:
    """Mid path for the move-flag encoding.

    The mid only moves on steps with moved >= 0.5, and then by at least one
    half-tick. It is a loop because each move depends on the previous mid.
    """
    mid, path = float(prev_mid), []
    for r, flag in zip(log_returns.tolist(), moved.tolist()):
        if flag >= 0.5:
            move = mid * math.expm1(r)
            if round_to_tick:
                move = math.copysign(max(1, round(abs(move) / HALF_TICK)) * HALF_TICK, move)
            mid += move
        path.append(mid)
    return torch.tensor(path, dtype=torch.float64)


def _parity_spread(spread_ticks: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    """Round spreads to the nearest tick count the mid allows (at least 1).

    Bid and ask sit on the 1-cent grid, so a mid on a whole tick needs an even
    spread and a mid on a half tick an odd one. Rounding freely would shift
    the book by half a tick and create mid moves on steps flagged as unmoved.
    """
    parity = (torch.round(mid / HALF_TICK) % 2).double()     # 1 when the mid is on a half tick
    nearest = 2 * torch.round((spread_ticks - parity) / 2) + parity
    return torch.where(nearest < 1, nearest + 2, nearest)


def decode_book(
    features: torch.Tensor,
    prev_mid: torch.Tensor | float,
    representation: str = "structured",
    levels: int = 10,
    round_to_tick: bool = True,
    move_flag: bool = False,
) -> OrderBook:
    """Inverse of encode_book for one sequence of shape (T, F).

    prev_mid is the mid just before the first row. No constraint is enforced:
    crossed books, negative sizes and repeated levels are kept for the audit.
    """
    f = features.double()
    if f.dim() != 2 or f.shape[1] != len(feature_names(representation, levels, move_flag)):
        raise ValueError("features must have shape (T, F) for the given representation")
    if move_flag:
        mid = _flagged_mid_path(f[:, 0], f[:, 1], float(prev_mid), round_to_tick)
        f = torch.cat([f[:, :1], f[:, 2:]], dim=1)     # drop the flag column
    else:
        mid = torch.as_tensor(prev_mid, dtype=torch.float64) * torch.exp(torch.cumsum(f[:, 0], dim=0))

    if representation == "structured":
        spread = torch.exp(f[:, 1])
        ask_gaps = torch.exp(f[:, 2:levels + 1])
        bid_gaps = torch.exp(f[:, levels + 1:2 * levels])
        if round_to_tick:
            ask_gaps, bid_gaps = torch.round(ask_gaps), torch.round(bid_gaps)
            spread = _parity_spread(spread, mid) if move_flag else torch.round(spread)
        best_bid = mid - spread * TICK / 2
        if round_to_tick:
            best_bid = _snap(best_bid)
        best_ask = best_bid + spread * TICK
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


# ---------------------------------------------------------------------------
# Scaling and windowing
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Per-feature scaling fitted on the training period only.

    "minmax" maps training values to [0, 1], matching the recovery network's
    sigmoid output; "standard" gives zero mean and unit variance.
    """

    METHODS = ("minmax", "standard")

    def __init__(self, method: str = "minmax", eps: float = 1e-8):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self.eps = eps
        self.shift: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> FeatureScaler:
        x = x.double()
        if self.method == "minmax":
            self.shift = x.amin(dim=0)
            self.scale = x.amax(dim=0) - self.shift
        else:
            self.shift = x.mean(dim=0)
            self.scale = x.std(dim=0)
        self.scale = self.scale.clamp_min(self.eps)
        return self

    def _check_fitted(self) -> None:
        if self.shift is None:
            raise RuntimeError("FeatureScaler must be fitted first")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return (x.double() - self.shift) / self.scale

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return x.double() * self.scale + self.shift

    def state_dict(self) -> dict:
        self._check_fitted()
        return {"method": self.method, "eps": self.eps, "shift": self.shift, "scale": self.scale}

    @classmethod
    def from_state_dict(cls, state: dict) -> FeatureScaler:
        scaler = cls(state["method"], state["eps"])
        scaler.shift, scaler.scale = state["shift"], state["scale"]
        return scaler


class LOBWindowDataset(Dataset):
    """Fixed-length float32 windows over one period.

    Windows are views into one tensor, so stride 1 costs no extra memory.
    prev_mid[j] is the mid-price just before feature row j.
    """

    def __init__(self, features: torch.Tensor, prev_mid: torch.Tensor, time: torch.Tensor,
                 seq_len: int, stride: int = 1):
        if not len(features) == len(prev_mid) == len(time):
            raise ValueError("features, prev_mid and time must have the same length")
        if len(features) < seq_len:
            raise ValueError(f"Period has {len(features)} rows, fewer than seq_len={seq_len}")
        self.features = features.float().contiguous()
        self.prev_mid = prev_mid.double()
        self.time = time.double()
        self.seq_len = seq_len
        self.stride = stride
        self.starts = torch.arange(0, len(features) - seq_len + 1, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> torch.Tensor:
        s = int(self.starts[i])
        return self.features[s:s + self.seq_len]

    def anchor(self, i: int) -> float:
        """Mid-price just before window i."""
        return float(self.prev_mid[int(self.starts[i])])

    def window_time(self, i: int) -> torch.Tensor:
        s = int(self.starts[i])
        return self.time[s:s + self.seq_len]

    def stack(self, indices: Iterable[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows (n, T, F) and their anchors (n,), all windows by default."""
        idx = list(range(len(self)) if indices is None else indices)
        windows = torch.stack([self[i] for i in idx])
        anchors = torch.tensor([self.anchor(i) for i in idx], dtype=torch.float64)
        return windows, anchors

    def disjoint_indices(self) -> range:
        """Non-overlapping windows, used for reference sets taken from training data."""
        return range(0, len(self), self.seq_len)


@dataclass
class LOBSplits:
    train: LOBWindowDataset
    val: LOBWindowDataset
    test: LOBWindowDataset
    scaler: FeatureScaler
    feature_names: list[str]
    config: dict          # the build_datasets arguments

    @property
    def representation(self) -> str:
        return self.config["representation"]

    @property
    def move_flag(self) -> bool:
        return self.config["move_flag"]


def build_datasets(
    data_dir: str | Path,
    seq_len: int = 64,
    representation: str = "structured",
    event_stride: int = 10,
    scaling: str = "minmax",
    open_minutes: float = 10,
    close_minutes: float = 10,
    train_end: str = "14:00",
    val_end: str = "15:00",
    train_stride: int = 1,
    eval_stride: int | None = None,
    move_flag: bool = False,
) -> LOBSplits:
    """Run the whole pipeline and return the three windowed periods.

    Encoding happens before the split. Each feature only looks one step back,
    so the first validation return uses the last training mid, which is past
    data. The scaler sees training rows only, and evaluation windows do not
    overlap by default (eval_stride = seq_len).
    """
    config = dict(locals())
    config["data_dir"] = str(data_dir)
    config["eval_stride"] = eval_stride = eval_stride or seq_len

    book = subsample_events(trim_session(load_lobster(data_dir), open_minutes, close_minutes), event_stride)
    features = encode_book(book, representation, move_flag)
    prev_mid = book.mid[:-1]
    time = book.time[1:]
    parts = split_by_time(time, train_end, val_end)

    scaler = FeatureScaler(scaling).fit(features[parts["train"]])
    scaled = scaler.transform(features)

    def window(name: str, stride: int) -> LOBWindowDataset:
        s = parts[name]
        return LOBWindowDataset(scaled[s], prev_mid[s], time[s], seq_len, stride)

    return LOBSplits(
        train=window("train", train_stride),
        val=window("val", eval_stride),
        test=window("test", eval_stride),
        scaler=scaler,
        feature_names=feature_names(representation, book.levels, move_flag),
        config=config,
    )
