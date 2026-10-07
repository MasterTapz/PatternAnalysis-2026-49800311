"""Train TimeGAN on LOBSTER order book windows.

Runs the three training phases of Yoon et al. (2019):
  1. autoencoder  embedder and recovery learn a latent space for real windows
  2. supervisor   the supervisor learns to predict the next latent step
  3. joint        generator, supervisor, embedder, recovery and discriminator
                  train together, two generator updates per discriminator one
The no-supervisor ablation (--no-supervisor) skips phase 2 and drops every
supervised term, so comparing the two runs isolates what that loss adds.
The baseline (--model rgan) is a plain recurrent GAN in feature space; it
skips phases 1 and 2 and trains in the joint phase with one generator and
one discriminator update per step.

Validation uses the 14:00 to 15:00 hour only. During phase 3, generated
samples are decoded back into order books and scored against the validation
hour and against a reference set of training windows; the checkpoint with the
lowest validation score is kept as best.pt. The test hour is only used once,
after training, to score best.pt.

Examples (from the project folder, the one holding src/):
  python src/train.py --data-dir ../../../../data/LOBSTER
  python src/train.py --data-dir ../../../../data/LOBSTER --no-supervisor
  python src/train.py --data-dir ../../../../data/LOBSTER --model rgan
  python src/train.py --data-dir ../../../../data/LOBSTER --representation raw
  python src/train.py --data-dir ../../../../data/LOBSTER --run-name structured_s0 --resume
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import LOBSplits, LOBWindowDataset, build_datasets
from helpers.checkpoints import (BEST_CHECKPOINT, LAST_CHECKPOINT, MODELS, PHASES, build_model, default_run_name,
                                 load_checkpoint, restore_rng_state, resumed_args, rng_state, run_state)
from helpers.evaluation import Evaluator, evaluate_test
from helpers.plotting import plot_history
from modules import TimeGAN


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = p.add_argument_group("data")
    data.add_argument("--data-dir", required=True, help="Folder holding the LOBSTER CSVs")
    data.add_argument("--representation", choices=("structured", "raw"), default="structured")
    data.add_argument("--scaling", choices=("minmax", "standard"), default="minmax")
    data.add_argument("--seq-len", type=int, default=64)
    data.add_argument("--event-stride", type=int, default=10)
    data.add_argument("--move-flag", action="store_true",
                      help="Add a binary mid-moved feature so zero price moves can be generated")

    model = p.add_argument_group("model")
    model.add_argument("--model", choices=tuple(MODELS), default="timegan",
                       help="timegan, or rgan for the baseline recurrent GAN (one adversarial phase only)")
    model.add_argument("--baseline-moment-weight", type=float, default=0.0,
                       help="Optional moment loss weight for the rgan baseline (0 = pure adversarial)")
    model.add_argument("--hidden-dim", type=int, default=64)
    model.add_argument("--num-layers", type=int, default=3)
    model.add_argument("--no-supervisor", action="store_true", help="Ablation without the supervised loss")
    model.add_argument("--paper-weights", action="store_true",
                       help="Loss weights as written in the paper instead of the authors' code")

    train = p.add_argument_group("training")
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--ae-steps", type=int, default=10_000)
    train.add_argument("--sup-steps", type=int, default=10_000)
    train.add_argument("--joint-steps", type=int, default=10_000)
    train.add_argument("--clip-grad", type=float, default=None, help="Max gradient norm (off by default)")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    run = p.add_argument_group("logging and checkpoints")
    run.add_argument("--out-dir", default="runs")
    run.add_argument("--run-name", default=None, help="Defaults to <representation>_<variant>_s<seed>")
    run.add_argument("--log-every", type=int, default=100)
    run.add_argument("--eval-every", type=int, default=500)
    run.add_argument("--eval-samples", type=int, default=512)
    run.add_argument("--resume", action="store_true", help="Continue from <run>/last.pt")
    args = p.parse_args(argv)
    if args.model == "rgan" and args.no_supervisor:
        p.error("--no-supervisor only applies to --model timegan")
    if args.run_name is None:
        args.run_name = default_run_name(args)
    return args


def batches(dataset: LOBWindowDataset, batch_size: int, seed: int, device: str) -> Iterator[torch.Tensor]:
    """Endless stream of shuffled training batches on the target device.

    Shuffling only changes which training windows share a batch; every window
    still comes from the training hours and keeps its internal time order.
    """
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
    while True:
        for x in loader:
            yield x.to(device, non_blocking=True)


def grad_norm(params: list[torch.nn.Parameter], clip: float | None) -> float:
    """Total gradient norm before clipping; clips in place when clip is set."""
    return float(torch.nn.utils.clip_grad_norm_(params, clip if clip is not None else float("inf")))


def as_floats(logs: dict[str, torch.Tensor], prefix: str = "") -> dict[str, float]:
    """Detached loss components as plain floats for the history, keys optionally prefixed."""
    return {f"{prefix}{k}": float(v) for k, v in logs.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class Trainer:
    """Owns the model, optimisers, history and checkpoints for one run."""

    def __init__(self, args: argparse.Namespace, splits: LOBSplits, run_dir: Path):
        self.args, self.splits, self.run_dir = args, splits, run_dir
        self.device = args.device
        self.model = build_model(args, len(splits.feature_names)).to(self.device)
        self.is_timegan = isinstance(self.model, TimeGAN)
        self.optimisers = self._build_optimisers()
        self.evaluator = Evaluator(splits, self.device, args.eval_samples, args.seed)
        self.data = batches(splits.train, args.batch_size, args.seed, self.device)
        self.history = {"train": [], "val": [], "phases": {}}
        self.phase_index, self.step, self.best_score = 0, 0, float("inf")
        self._mark = (time.time(), 0)   # (wall clock, step) where the current timing interval started

    def _build_optimisers(self) -> dict[str, torch.optim.Optimizer]:
        """One Adam optimiser per group of networks that a loss trains."""
        def adam(params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
            return torch.optim.Adam(params, lr=self.args.lr)

        optimisers = {
            "generator": adam(self.model.generator_parameters()),
            "discriminator": adam(self.model.discriminator_parameters()),
        }
        if self.is_timegan:
            optimisers["autoencoder"] = adam(self.model.autoencoder_parameters())
            if self.model.supervisor is not None:
                optimisers["supervisor"] = adam(self.model.supervisor_parameters())
        return optimisers

    # -- checkpoints -----------------------------------------------------------

    def save(self) -> None:
        """Write last.pt (everything --resume needs) and history.json."""
        state = run_state(self.model, self.splits, self.args)
        state.update({
            "optimisers": {k: o.state_dict() for k, o in self.optimisers.items()},
            "phase_index": self.phase_index, "step": self.step, "best_score": self.best_score,
            "history": self.history, **rng_state(),
        })
        torch.save(state, self.run_dir / LAST_CHECKPOINT)
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=1))

    def save_best(self, score: float) -> None:
        state = run_state(self.model, self.splits, self.args)
        state.update({"phase_index": self.phase_index, "step": self.step, "score": score})
        torch.save(state, self.run_dir / BEST_CHECKPOINT)

    def load(self) -> None:
        """Restore model, optimisers, progress, history and random states from last.pt."""
        state = load_checkpoint(self.run_dir / LAST_CHECKPOINT, self.device)
        self.model.load_state_dict(state["model"])
        for k, o in self.optimisers.items():
            o.load_state_dict(state["optimisers"][k])
        self.phase_index, self.step, self.best_score = state["phase_index"], state["step"], state["best_score"]
        self.history = state["history"]
        restore_rng_state(state)
        print(f"Resumed {self.run_dir.name} at phase {PHASES[self.phase_index]}, step {self.step}")

    # -- logging ---------------------------------------------------------------

    def _log_train(self, phase: str, window: list[dict]) -> None:
        mean = {k: sum(d[k] for d in window) / len(window) for k in window[0]}
        self.history["train"].append({"phase": phase, "step": self.step, **mean})

    def _log_val(self, phase: str, metrics: dict) -> None:
        self.history["val"].append({"phase": phase, "step": self.step, **metrics})
        shown = {k: round(v, 4) for k, v in metrics.items() if not k.endswith("reverse")}
        print(f"  [{phase} {self.step:6d}] {shown}", flush=True)

    # -- one optimisation step per phase ---------------------------------------

    def _update(self, name: str, loss: torch.Tensor) -> float:
        """Backpropagate loss and step optimiser name; returns the gradient norm before clipping."""
        opt = self.optimisers[name]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = grad_norm([p for g in opt.param_groups for p in g["params"]], self.args.clip_grad)
        opt.step()
        return norm

    def _autoencoder_step(self) -> dict:
        loss, logs = self.model.autoencoder_loss(next(self.data))
        self._update("autoencoder", loss)
        return as_floats(logs)

    def _supervisor_step(self) -> dict:
        loss, logs = self.model.supervisor_loss(next(self.data))
        self._update("supervisor", loss)
        return as_floats(logs)

    def _joint_step(self) -> dict:
        return self._timegan_joint_step() if self.is_timegan else self._baseline_step()

    def _timegan_joint_step(self) -> dict:
        """Two generator and embedder updates, then one discriminator update unless its loss is already low."""
        out: dict[str, float] = {}
        for _ in range(2):   # two generator and embedder updates per discriminator update, as in the authors' code
            x = next(self.data)
            g_loss, g_logs = self.model.generator_loss(x)
            out["g_grad_norm"] = self._update("generator", g_loss)
            e_loss, e_logs = self.model.autoencoder_loss(x, with_supervised=True)
            self._update("autoencoder", e_loss)
        d_loss, d_logs = self.model.discriminator_loss(next(self.data))
        update = self.model.should_update_discriminator(d_loss)
        out["d_grad_norm"] = self._update("discriminator", d_loss) if update else 0.0
        out["d_updated"] = float(update)
        return {**out, **as_floats(g_logs, "g_"), **as_floats(e_logs, "e_"), **as_floats(d_logs, "d_")}

    def _baseline_step(self) -> dict:
        """One generator and one discriminator update for the baseline recurrent GAN."""
        g_loss, g_logs = self.model.generator_loss(next(self.data))
        out = {"g_grad_norm": self._update("generator", g_loss)}
        d_loss, d_logs = self.model.discriminator_loss(next(self.data))
        out["d_grad_norm"] = self._update("discriminator", d_loss)
        out["d_updated"] = 1.0
        return {**out, **as_floats(g_logs, "g_"), **as_floats(d_logs, "d_")}

    # -- phase driver ------------------------------------------------------------

    def run(self) -> None:
        """Run the remaining phases in order, saving last.pt after each one."""
        steps = {"autoencoder": self.args.ae_steps, "supervisor": self.args.sup_steps, "joint": self.args.joint_steps}
        step_fns = {"autoencoder": self._autoencoder_step, "supervisor": self._supervisor_step, "joint": self._joint_step}
        while self.phase_index < len(PHASES):
            phase = PHASES[self.phase_index]
            reason = self._skip_reason(phase)
            if reason:
                print(f"Phase {phase}: skipped, {reason}")
                self._next_phase()
                continue
            self._run_phase(phase, steps[phase], step_fns[phase])
            self._next_phase()
            self.save()

    def _skip_reason(self, phase: str) -> str | None:
        """Why this model has no such phase, or None when it should run."""
        if not self.is_timegan and phase != "joint":
            return "the baseline only has the adversarial phase"
        if phase == "supervisor" and self.model.supervisor is None:
            return "the ablation has no supervisor"
        return None

    def _next_phase(self) -> None:
        self.phase_index, self.step = self.phase_index + 1, 0

    def _run_phase(self, phase: str, total: int, step_fn: Callable[[], dict]) -> None:
        print(f"Phase {phase}: steps {self.step} to {total}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start, first, window = time.time(), self.step, []
        self._mark = (time.time(), self.step)
        self.model.train()
        while self.step < total:
            window.append(step_fn())
            self.step += 1
            if self.step % self.args.log_every == 0 or self.step == total:
                self._log_train(phase, window)
                window = []
            if self.step % self.args.eval_every == 0 or self.step == total:
                self._account(phase)
                self._evaluate(phase)
        print(f"Phase {phase} finished: {self.step - first} steps in {time.time() - start:.0f} s", flush=True)

    def _account(self, phase: str) -> None:
        """Add the training time and steps since the last mark to the phase record.

        Called before every evaluation (and so before every checkpoint), so an
        interrupted run keeps its timing; evaluation time is not counted.
        """
        then, step = self._mark
        record = self.history["phases"].setdefault(phase, {"seconds": 0.0, "steps": 0})
        record["seconds"] += time.time() - then
        record["steps"] += self.step - step
        record["steps_per_second"] = record["steps"] / max(record["seconds"], 1e-9)
        record["sequences_per_second"] = record["steps_per_second"] * self.args.batch_size
        if torch.cuda.is_available():
            record["peak_vram_mib"] = max(record.get("peak_vram_mib", 0.0), torch.cuda.max_memory_allocated() / 2**20)

    def _evaluate(self, phase: str) -> None:
        """Score on the validation hour, keep best.pt in the joint phase, then save last.pt."""
        self.model.eval()
        metrics = self.evaluator.reconstruction(self.model)
        if phase == "joint":
            metrics.update(self.evaluator.samples(self.model))
            if metrics["score"] < self.best_score:
                self.best_score = metrics["score"]
                self.save_best(metrics["score"])
                metrics["new_best"] = 1.0
        self._log_val(phase, metrics)
        self.model.train()
        self.save()
        self._mark = (time.time(), self.step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    run_dir = Path(args.out_dir) / args.run_name
    if args.resume:
        args = resumed_args(run_dir, args.device)
    elif (run_dir / LAST_CHECKPOINT).exists():
        raise FileExistsError(f"{run_dir} already has a run; pass --resume or choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    splits = build_datasets(args.data_dir, seq_len=args.seq_len, representation=args.representation,
                            event_stride=args.event_stride, scaling=args.scaling,
                            move_flag=getattr(args, "move_flag", False))   # runs from before the flag have none
    trainer = Trainer(args, splits, run_dir)
    if args.resume:
        trainer.load()
    print(f"Run {run_dir} on {args.device}: {len(splits.train)} training windows, "
          f"parameters {trainer.model.parameter_counts()}", flush=True)
    (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "model": asdict(trainer.model.config),
                                                     "data": splits.config}, indent=1))
    trainer.run()
    plot_history(trainer.history, run_dir / "training_curves.png")
    print(f"Saved {run_dir / 'training_curves.png'}")
    evaluate_test(run_dir, splits, args.device, args.seed)
    return run_dir


if __name__ == "__main__":
    main()
