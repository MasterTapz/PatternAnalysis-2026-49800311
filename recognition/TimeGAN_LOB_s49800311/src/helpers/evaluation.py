"""Validation scoring during training and the test-hour evaluation afterwards.

Evaluator only sees the validation hour (14:00 to 15:00) and its score
picks best.pt. evaluate_test runs once after training on best.pt and the
test hour (15:00 to 15:50), and writes test_metrics.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from dataset import LOBSplits
from helpers.checkpoints import PHASES, best_checkpoint, load_checkpoint, model_from_state
from helpers.metrics import (book_violations, decode_real_windows, decode_synthetic_windows, distribution_report,
                             kl_only)
from modules import SequenceGAN, TimeGAN


def reconstruction_mse(model: TimeGAN, x: torch.Tensor) -> float:
    return float(torch.mean((model.reconstruct(x) - x) ** 2))


class Evaluator:
    """Scores a model on the validation hour.

    The reference books are decoded once: every validation window and
    non-overlapping training windows from all training hours.
    """

    def __init__(self, splits: LOBSplits, device: str, n_samples: int, seed: int):
        self.splits, self.device, self.n_samples = splits, device, n_samples
        self.val_x, self.val_anchor, self.val_books = decode_real_windows(splits, splits.val)
        _, _, self.train_books = decode_real_windows(splits, splits.train, splits.train.disjoint_indices())
        self.generator = torch.Generator().manual_seed(seed)   # for anchors only, separate from training

    @torch.no_grad()
    def reconstruction(self, model: SequenceGAN) -> dict[str, float]:
        """Autoencoder and supervisor error on the validation windows (TimeGAN only)."""
        if not isinstance(model, TimeGAN):
            return {}
        x = self.val_x.to(self.device)
        out = {"val_recon_mse": reconstruction_mse(model, x)}
        if model.supervisor is not None:
            out["val_supervised_mse"] = float(model.supervisor_loss(x)[0])
        return out

    @torch.no_grad()
    def samples(self, model: SequenceGAN) -> dict[str, float]:
        """Validity and KL of synthetic books against validation and training data.

        score = validation spread KL + validation return KL + any-violation rate.
        """
        fake_x = model.sample(self.n_samples, self.splits.val.seq_len, self.device)
        fake = decode_synthetic_windows(self.splits, fake_x, self.val_anchor, self.generator)
        out = book_violations(fake)
        out.update({f"val_{k}": v for k, v in kl_only(distribution_report(self.val_books, fake)).items()})
        out.update({f"train_{k}": v for k, v in kl_only(distribution_report(self.train_books, fake)).items()})
        out["score"] = out["val_kl_spread"] + out["val_kl_return"] + out["any_violation_rate"]
        return out


def timed_sample(model: SequenceGAN, n: int, seq_len: int, device: str) -> tuple[torch.Tensor, float]:
    """n synthetic windows and the seconds it took, waiting for the GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    fake_x = model.sample(n, seq_len, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return fake_x, time.time() - start


@torch.no_grad()
def evaluate_test(run_dir: Path, splits: LOBSplits, device: str, seed: int, n_samples: int = 1024) -> dict:
    """Score best.pt on the test hour and write test_metrics.json.

    The real-vs-real KL floors are included, since no generator can be
    expected to beat them. Sampling uses seed + 1, so the numbers differ
    slightly from predict.py, which uses seed.
    """
    checkpoint = best_checkpoint(run_dir)
    state = load_checkpoint(checkpoint, device)
    model = model_from_state(state, device)

    references = Evaluator(splits, device, n_samples, seed)
    test_x, test_anchor, test_books = decode_real_windows(splits, splits.test)

    torch.manual_seed(seed + 1)
    fake_x, sample_seconds = timed_sample(model, n_samples, splits.test.seq_len, device)
    fake = decode_synthetic_windows(splits, fake_x, test_anchor, torch.Generator().manual_seed(seed))

    report = {
        "checkpoint": checkpoint.name,
        "checkpoint_phase_step": [PHASES[min(state["phase_index"], 2)], state["step"]],
        "test_recon_mse": reconstruction_mse(model, test_x.to(device)) if isinstance(model, TimeGAN) else None,
        "violations": book_violations(fake),
        "vs_test_hour": distribution_report(test_books, fake),
        "vs_training_reference": distribution_report(references.train_books, fake),
        "real_floor_training_vs_test": distribution_report(test_books, references.train_books),
        "real_floor_validation_vs_test": distribution_report(test_books, references.val_books),
        "parameters": model.parameter_counts(),
        "sampling": {"sequences": n_samples, "seconds": sample_seconds,
                     "sequences_per_second": n_samples / max(sample_seconds, 1e-9)},
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(report, indent=1))
    print_test_report(report, n_real=len(test_books))
    return report


def print_test_report(report: dict, n_real: int) -> None:
    n_samples = report["sampling"]["sequences"]
    print(f"\nTest hour evaluation of {report['checkpoint']} ({n_samples} synthetic windows vs {n_real} real)")
    print(f"  invalid books: {report['violations']}")
    rows = [("synthetic vs test hour", "vs_test_hour"),
            ("synthetic vs training reference", "vs_training_reference"),
            ("real floor: training vs test", "real_floor_training_vs_test"),
            ("real floor: validation vs test", "real_floor_validation_vs_test")]
    for label, key in rows:
        print(f"  {label:32s} spread KL {report[key]['kl_spread']:7.3f}   return KL {report[key]['kl_return']:7.3f}")
    print(f"  sampling: {report['sampling']['sequences_per_second']:.0f} sequences per second")
