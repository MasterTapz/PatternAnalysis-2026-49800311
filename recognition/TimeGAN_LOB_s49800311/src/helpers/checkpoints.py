"""Model registry, run names and checkpoint files.

A checkpoint is a dict saved with torch.save. Both files hold the model
weights, model type and config, the fitted scaler, the data pipeline
arguments, the feature names and the train.py arguments, so a run can be
rebuilt without its command line. last.pt adds what --resume needs
(optimisers, progress, history, random states); best.pt adds the step and
validation score it was kept at.
"""
from __future__ import annotations

import argparse
import inspect
from dataclasses import asdict
from pathlib import Path

import torch

from dataset import LOBSplits, build_datasets
from modules import (LadderTicks, RawBookPenalty, RecurrentGAN, RecurrentGANConfig, SequenceGAN, TimeGAN,
                     TimeGANConfig)

PHASES = ("autoencoder", "supervisor", "joint")
MODELS = {"timegan": (TimeGAN, TimeGANConfig), "rgan": (RecurrentGAN, RecurrentGANConfig)}
LAST_CHECKPOINT, BEST_CHECKPOINT = "last.pt", "best.pt"


# ---------------------------------------------------------------------------
# Building models
# ---------------------------------------------------------------------------

def default_run_name(args: argparse.Namespace) -> str:
    """[<data>_]<representation>_<variant>[_paper][_std][_mf][_hf][_ema][_sn<k>][_dw<w>][_cc<w>][_tw<w>][_bw<w>]_s<seed>.

    data is <ticker>_<clock>_T<seq_len>[_v<eval stride>], with clock 1s for a
    one-second clock or e10 for every 10th event, e.g. INTC_1s_T504_v24. It is
    left out for the original setup (AMZN, e10, T64), so the older run folders
    keep their names. variant is timegan, nosup (no supervisor), rgan (plain
    baseline) or rganm (baseline with the moment loss). Every option that
    changes the results gets a suffix, so two different runs never share a folder.
    """
    clock = f"{args.clock_seconds:g}s" if args.clock_seconds > 0 else f"e{args.event_stride}"
    data = f"{args.ticker}_{clock}_T{args.seq_len}"
    data += f"_v{args.eval_stride}" if args.eval_stride else ""
    prefix = "" if data == "AMZN_e10_T64" else f"{data}_"
    if args.model == "rgan":
        variant = "rganm" if args.baseline_moment_weight > 0 else "rgan"
    else:
        variant = "nosup" if args.no_supervisor else "timegan"
    suffix = "_paper" if args.paper_weights and args.model == "timegan" else ""
    suffix += "_std" if args.scaling == "standard" else ""
    suffix += "_mf" if args.move_flag else ""
    suffix += "_hf" if args.hold_flags else ""
    suffix += "_ema" if args.ema_decay > 0 else ""
    suffix += f"_sn{args.static_noise_dim}" if args.static_noise_dim > 0 and args.model == "timegan" else ""
    suffix += f"_dw{args.difference_weight:g}" if args.difference_weight > 0 and args.model == "timegan" else ""
    suffix += f"_cc{args.change_corr_weight:g}" if args.change_corr_weight > 0 and args.model == "timegan" else ""
    suffix += f"_tw{args.tick_weight:g}" if args.tick_weight > 0 and args.model == "timegan" else ""
    suffix += f"_bw{args.book_weight:g}" if args.book_weight > 0 and args.model == "timegan" else ""
    return f"{prefix}{args.representation}_{variant}{suffix}_s{args.seed}"


def ladder_decoder(splits: LOBSplits) -> LadderTicks | None:
    """The tick-space decoder behind L_P, for the structured encoding only."""
    if splits.representation != "structured":
        return None
    names = splits.feature_names

    def columns(prefix: str) -> list[int]:
        return [i for i, name in enumerate(names) if name.startswith(prefix)]

    return LadderTicks(splits.scaler.shift, splits.scaler.scale, columns("log_ask_gap"), columns("log_bid_gap"),
                       columns("log1p_ask_size"), columns("log1p_bid_size"))


def book_penalty(splits: LOBSplits) -> RawBookPenalty | None:
    """The hinge penalties behind L_book, for the raw encoding only."""
    if splits.representation != "raw":
        return None
    names = splits.feature_names
    return RawBookPenalty(splits.scaler.shift, splits.scaler.scale,
                          [names.index(f"ask_offset_{k}") for k in range(1, 11)],
                          [names.index(f"bid_offset_{k}") for k in range(1, 11)])


def build_model(args: argparse.Namespace, feature_dim: int, ladder: LadderTicks | None = None,
                book: RawBookPenalty | None = None) -> SequenceGAN:
    """TimeGAN, its ablation or the baseline, as set on the command line.

    ladder is the decoder TimeGAN needs for L_P (see ladder_decoder) and
    book the penalty behind L_book (see book_penalty).
    """
    activation = "sigmoid" if args.scaling == "minmax" else "identity"
    if args.model == "rgan":
        return RecurrentGAN(RecurrentGANConfig(
            feature_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            output_activation=activation, moment_weight=args.baseline_moment_weight))
    paper = dict(sqrt_losses=False, eta=10.0, recon_weight=1.0, embed_supervised_weight=1.0)
    return TimeGAN(TimeGANConfig(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        recovery_activation=activation,
        use_supervisor=not args.no_supervisor,
        static_noise_dim=args.static_noise_dim,
        difference_weight=args.difference_weight,
        change_corr_weight=args.change_corr_weight,
        tick_weight=args.tick_weight,
        book_weight=args.book_weight,
        **(paper if args.paper_weights else {}),
    ), ladder, book)


def model_type(model: SequenceGAN) -> str:
    """The MODELS key of a model."""
    return next(name for name, (model_cls, _) in MODELS.items() if isinstance(model, model_cls))


# ---------------------------------------------------------------------------
# Writing checkpoints
# ---------------------------------------------------------------------------

def run_state(model: SequenceGAN, splits: LOBSplits, args: argparse.Namespace) -> dict:
    """The part of every checkpoint that describes the model and its data."""
    return {
        "model": model.state_dict(),
        "model_type": model_type(model),
        "model_config": asdict(model.config),
        "scaler": splits.scaler.state_dict(),
        "data_config": splits.config,
        "feature_names": splits.feature_names,
        "args": vars(args),
    }


def rng_state() -> dict:
    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    # Loading with map_location moves every tensor to the GPU, but RNG states must be CPU tensors.
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])


# ---------------------------------------------------------------------------
# Reading checkpoints
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path, device: str) -> dict:
    # weights_only=False because checkpoints also hold configs and history; only load your own runs.
    return torch.load(path, map_location=device, weights_only=False)


def best_checkpoint(run_dir: Path) -> Path:
    """best.pt, or last.pt for a run that stopped before its first joint-phase validation."""
    best = run_dir / BEST_CHECKPOINT
    return best if best.exists() else run_dir / LAST_CHECKPOINT


def resumed_args(run_dir: Path, defaults: argparse.Namespace, device: str) -> argparse.Namespace:
    """The arguments a run was started with, for --resume.

    Options added to train.py after the run started take their defaults.
    Only the device may change.
    """
    saved = load_checkpoint(run_dir / LAST_CHECKPOINT, "cpu")["args"]
    return argparse.Namespace(**{**vars(defaults), **saved, "resume": True, "device": device})


def model_from_state(state: dict, device: str) -> SequenceGAN:
    """Rebuild a saved model in eval mode."""
    model_cls, config_cls = MODELS[state.get("model_type", "timegan")]
    model = model_cls(config_cls(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def splits_from_state(state: dict, data_dir: str) -> LOBSplits:
    """Rebuild the data pipeline a checkpoint was trained on.

    Settings the checkpoint does not have (such as move_flag in older runs)
    keep the build_datasets defaults.
    """
    wanted = inspect.signature(build_datasets).parameters
    config = {k: v for k, v in state["data_config"].items() if k in wanted}
    config["data_dir"] = data_dir
    return build_datasets(**config)


def load_run(run_dir: Path, data_dir: str, device: str) -> tuple[SequenceGAN, LOBSplits, dict]:
    """The best model of a run, its data splits and the raw checkpoint."""
    state = load_checkpoint(best_checkpoint(run_dir), device)
    splits = splits_from_state(state, data_dir)
    return model_from_state(state, device), splits, state
