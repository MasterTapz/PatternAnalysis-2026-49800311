"""Model registry, run names and checkpoint files.

A checkpoint is a plain dict saved with torch.save. Both kinds hold enough to
rebuild the run without the original command line:
    model           the model's state_dict
    model_type      key into MODELS ("timegan" or "rgan")
    model_config    the model's config dataclass as a dict
    scaler          the training FeatureScaler's state_dict
    data_config     the build_datasets arguments, so the data pipeline is rebuilt exactly
    feature_names   names of the feature columns
    args            the train.py command line
last.pt adds the optimiser states, phase and step, best score, history and
random states that --resume needs; best.pt adds the phase, step and
validation score it was kept at. Runs saved before the move flag existed
have no move_flag entry anywhere, and every reader treats it as False.
"""
from __future__ import annotations

import argparse
import inspect
from dataclasses import asdict
from pathlib import Path

import torch

from dataset import LOBSplits, build_datasets
from modules import RecurrentGAN, RecurrentGANConfig, SequenceGAN, TimeGAN, TimeGANConfig

PHASES = ("autoencoder", "supervisor", "joint")   # a checkpoint's phase_index points into this tuple
MODELS = {"timegan": (TimeGAN, TimeGANConfig), "rgan": (RecurrentGAN, RecurrentGANConfig)}
LAST_CHECKPOINT, BEST_CHECKPOINT = "last.pt", "best.pt"


# ---------------------------------------------------------------------------
# Building models
# ---------------------------------------------------------------------------

def default_run_name(args: argparse.Namespace) -> str:
    """<representation>_<variant>[_paper][_std][_mf]_s<seed>, unique for every setting that changes results.

    variant is timegan, nosup (no supervisor), rgan (plain baseline) or rganm
    (baseline with the moment loss). _paper marks the paper's loss weights,
    _std standard scaling and _mf the move-flag encoding, so these runs never
    clash with the defaults.
    """
    if args.model == "rgan":
        variant = "rganm" if args.baseline_moment_weight > 0 else "rgan"
    else:
        variant = "nosup" if args.no_supervisor else "timegan"
    suffix = ("_paper" if args.paper_weights and args.model == "timegan" else "")
    suffix += "_std" if args.scaling == "standard" else ""
    suffix += "_mf" if getattr(args, "move_flag", False) else ""
    return f"{args.representation}_{variant}{suffix}_s{args.seed}"


def build_model(args: argparse.Namespace, feature_dim: int) -> SequenceGAN:
    """The model implied by the command line: TimeGAN (or its ablation) or the baseline."""
    activation = "sigmoid" if args.scaling == "minmax" else "identity"
    if getattr(args, "model", "timegan") == "rgan":   # runs from before the baseline existed have no --model
        return RecurrentGAN(RecurrentGANConfig(
            feature_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            output_activation=activation, moment_weight=args.baseline_moment_weight))
    extra = dict(sqrt_losses=False, eta=10.0, recon_weight=1.0, embed_supervised_weight=1.0) if args.paper_weights else {}
    return TimeGAN(TimeGANConfig(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        recovery_activation=activation,
        use_supervisor=not args.no_supervisor,
        **extra,
    ))


def model_type(model: SequenceGAN) -> str:
    """The MODELS key of a model instance, stored in every checkpoint."""
    return next(name for name, (model_cls, _) in MODELS.items() if isinstance(model, model_cls))


# ---------------------------------------------------------------------------
# Writing checkpoints
# ---------------------------------------------------------------------------

def run_state(model: SequenceGAN, splits: LOBSplits, args: argparse.Namespace) -> dict:
    """The part of every checkpoint that describes the model and its data pipeline."""
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
    """CPU and CUDA random states, saved in last.pt so a resumed run draws the same numbers."""
    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Inverse of rng_state for a loaded checkpoint."""
    # map_location moves every tensor to the device, but RNG states must be CPU byte tensors.
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])


# ---------------------------------------------------------------------------
# Reading checkpoints
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path, device: str) -> dict:
    """A checkpoint dict with its tensors on device.

    weights_only=False because checkpoints also hold plain Python objects
    (configs, history, argument dicts); only load runs you produced yourself.
    """
    return torch.load(path, map_location=device, weights_only=False)


def best_checkpoint(run_dir: Path) -> Path:
    """best.pt, or last.pt for a run that never reached the joint phase's first validation."""
    best = run_dir / BEST_CHECKPOINT
    return best if best.exists() else run_dir / LAST_CHECKPOINT


def resumed_args(run_dir: Path, device: str) -> argparse.Namespace:
    """The command line a run was started with, for --resume; only the device may change."""
    saved = load_checkpoint(run_dir / LAST_CHECKPOINT, "cpu")["args"]
    saved.update(resume=True, device=device)
    return argparse.Namespace(**saved)


def model_from_state(state: dict, device: str) -> SequenceGAN:
    """Rebuild a saved model (TimeGAN, ablation or baseline) from a checkpoint dict, in eval mode."""
    model_cls, config_cls = MODELS[state.get("model_type", "timegan")]
    model = model_cls(config_cls(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def splits_from_state(state: dict, data_dir: str) -> LOBSplits:
    """Rebuild the exact data pipeline a checkpoint was trained on, reading the CSVs from data_dir.

    Only the entries build_datasets accepts are passed on, and anything the
    checkpoint lacks (such as move_flag in older runs) keeps its default.
    """
    wanted = inspect.signature(build_datasets).parameters
    config = {k: v for k, v in state["data_config"].items() if k in wanted}
    config["data_dir"] = data_dir
    return build_datasets(**config)


def load_run(run_dir: Path, data_dir: str, device: str) -> tuple[SequenceGAN, LOBSplits, dict]:
    """The best model of a run, the data splits it was trained with, and the raw checkpoint dict."""
    state = load_checkpoint(best_checkpoint(run_dir), device)
    splits = splits_from_state(state, data_dir)
    return model_from_state(state, device), splits, state
