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
lowest validation score is kept as best.pt. The test hour is never touched here.

Examples (from this folder):
  python train.py --data-dir ../../../../data/LOBSTER
  python train.py --data-dir ../../../../data/LOBSTER --no-supervisor
  python train.py --data-dir ../../../../data/LOBSTER --model rgan
  python train.py --data-dir ../../../../data/LOBSTER --representation raw
  python train.py --data-dir ../../../../data/LOBSTER --run-name structured_s0 --resume
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import LOBSplits, LOBWindowDataset, build_datasets
from metrics import book_violations, decode_windows, distribution_report
from modules import RecurrentGAN, RecurrentGANConfig, TimeGAN, TimeGANConfig

PHASES = ("autoencoder", "supervisor", "joint")
MODELS = {"timegan": (TimeGAN, TimeGANConfig), "rgan": (RecurrentGAN, RecurrentGANConfig)}


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


def build_model(args: argparse.Namespace, feature_dim: int) -> TimeGAN | RecurrentGAN:
    """The model implied by the command line: TimeGAN (or its ablation) or the baseline."""
    activation = "sigmoid" if args.scaling == "minmax" else "identity"
    if getattr(args, "model", "timegan") == "rgan":
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


def model_from_state(state: dict, device: str) -> TimeGAN | RecurrentGAN:
    """Rebuild a saved model (TimeGAN, ablation or baseline) from a checkpoint dict."""
    model_cls, config_cls = MODELS[state.get("model_type", "timegan")]
    model = model_cls(config_cls(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def batches(dataset: LOBWindowDataset, batch_size: int, seed: int, device: str):
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


# ---------------------------------------------------------------------------
# Evaluation on the validation hour
# ---------------------------------------------------------------------------

class Evaluator:
    """Scores the model on the validation hour; built once per run.

    The reference books are decoded once: every validation window, and
    non-overlapping training windows spread over all training hours.
    """

    def __init__(self, splits: LOBSplits, representation: str, device: str, n_samples: int, seed: int):
        self.splits, self.representation, self.device, self.n_samples = splits, representation, device, n_samples
        self.val_x, self.val_anchor = self._stack(splits.val, range(len(splits.val)))
        train_idx = range(0, len(splits.train), splits.train.seq_len)
        train_x, train_anchor = self._stack(splits.train, train_idx)
        self.move_flag = splits.config.get("move_flag", False)
        self.val_books = decode_windows(self.val_x, splits.scaler, self.val_anchor, representation, self.move_flag)
        self.train_books = decode_windows(train_x, splits.scaler, train_anchor, representation, self.move_flag)
        self.generator = torch.Generator().manual_seed(seed)

    @staticmethod
    def _stack(ds: LOBWindowDataset, idx) -> tuple[torch.Tensor, torch.Tensor]:
        idx = list(idx)
        return torch.stack([ds[i] for i in idx]), torch.tensor([ds.anchor(i) for i in idx], dtype=torch.float64)

    @torch.no_grad()
    def reconstruction(self, model: TimeGAN | RecurrentGAN) -> dict[str, float]:
        """Autoencoder and supervisor error on the validation windows (TimeGAN only)."""
        if not isinstance(model, TimeGAN):
            return {}
        x = self.val_x.to(self.device)
        out = {"val_recon_mse": float(torch.mean((model.reconstruct(x) - x) ** 2))}
        if model.supervisor is not None:
            out["val_supervised_mse"] = float(model.supervisor_loss(x)[0])
        return out

    @torch.no_grad()
    def samples(self, model: TimeGAN) -> dict[str, float]:
        """Generate, decode and score synthetic books against validation and training references."""
        fake_x = model.sample(self.n_samples, self.splits.val.seq_len, self.device)
        pick = torch.randint(len(self.val_anchor), (self.n_samples,), generator=self.generator)
        fake = decode_windows(fake_x, self.splits.scaler, self.val_anchor[pick], self.representation, self.move_flag)
        out = book_violations(fake)
        out.update({f"val_{k}": v for k, v in distribution_report(self.val_books, fake).items() if k.startswith("kl")})
        out.update({f"train_{k}": v for k, v in distribution_report(self.train_books, fake).items() if k.startswith("kl")})
        out["score"] = out["val_kl_spread"] + out["val_kl_return"] + out["any_violation_rate"]
        return out


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
        adam = lambda params: torch.optim.Adam(params, lr=args.lr)
        self.optimisers = {
            "generator": adam(self.model.generator_parameters()),
            "discriminator": adam(self.model.discriminator_parameters()),
        }
        if self.is_timegan:
            self.optimisers["autoencoder"] = adam(self.model.autoencoder_parameters())
            if self.model.supervisor is not None:
                self.optimisers["supervisor"] = adam(self.model.supervisor_parameters())
        self.evaluator = Evaluator(splits, args.representation, self.device, args.eval_samples, args.seed)
        self.data = batches(splits.train, args.batch_size, args.seed, self.device)
        self.history = {"train": [], "val": [], "phases": {}}
        self.phase_index, self.step, self.best_score = 0, 0, float("inf")

    # -- checkpoints -----------------------------------------------------------

    def _state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "model_type": "timegan" if self.is_timegan else "rgan",
            "model_config": asdict(self.model.config),
            "scaler": self.splits.scaler.state_dict(),
            "data_config": self.splits.config,
            "feature_names": self.splits.feature_names,
            "args": vars(self.args),
        }

    def save(self) -> None:
        state = self._state()
        state.update({
            "optimisers": {k: o.state_dict() for k, o in self.optimisers.items()},
            "phase_index": self.phase_index, "step": self.step, "best_score": self.best_score,
            "history": self.history, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })
        torch.save(state, self.run_dir / "last.pt")
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=1))

    def save_best(self, score: float) -> None:
        state = self._state()
        state.update({"phase_index": self.phase_index, "step": self.step, "score": score})
        torch.save(state, self.run_dir / "best.pt")

    def load(self) -> None:
        state = torch.load(self.run_dir / "last.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        for k, o in self.optimisers.items():
            o.load_state_dict(state["optimisers"][k])
        self.phase_index, self.step, self.best_score = state["phase_index"], state["step"], state["best_score"]
        self.history = state["history"]
        # map_location moves every tensor to the device, but RNG states must be CPU byte tensors.
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
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
        opt = self.optimisers[name]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = grad_norm([p for g in opt.param_groups for p in g["params"]], self.args.clip_grad)
        opt.step()
        return norm

    def _autoencoder_step(self) -> dict:
        loss, logs = self.model.autoencoder_loss(next(self.data))
        self._update("autoencoder", loss)
        return {k: float(v) for k, v in logs.items()}

    def _supervisor_step(self) -> dict:
        loss, logs = self.model.supervisor_loss(next(self.data))
        self._update("supervisor", loss)
        return {k: float(v) for k, v in logs.items()}

    def _baseline_step(self) -> dict:
        """One generator and one discriminator update for the baseline recurrent GAN."""
        g_loss, g_logs = self.model.generator_loss(next(self.data))
        out = {"g_grad_norm": self._update("generator", g_loss)}
        d_loss, d_logs = self.model.discriminator_loss(next(self.data))
        out["d_grad_norm"] = self._update("discriminator", d_loss)
        out["d_updated"] = 1.0
        out.update({f"g_{k}": float(v) for k, v in g_logs.items()})
        out.update({f"d_{k}": float(v) for k, v in d_logs.items()})
        return out

    def _joint_step(self) -> dict:
        if not self.is_timegan:
            return self._baseline_step()
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
        out.update({f"g_{k}": float(v) for k, v in g_logs.items()})
        out.update({f"e_{k}": float(v) for k, v in e_logs.items()})
        out.update({f"d_{k}": float(v) for k, v in d_logs.items()})
        return out

    # -- phase driver ------------------------------------------------------------

    def run(self) -> None:
        steps = {"autoencoder": self.args.ae_steps, "supervisor": self.args.sup_steps, "joint": self.args.joint_steps}
        step_fns = {"autoencoder": self._autoencoder_step, "supervisor": self._supervisor_step, "joint": self._joint_step}
        while self.phase_index < len(PHASES):
            phase = PHASES[self.phase_index]
            if not self.is_timegan and phase != "joint":
                print(f"Phase {phase}: skipped, the baseline only has the adversarial phase")
                self.phase_index, self.step = self.phase_index + 1, 0
                continue
            if phase == "supervisor" and self.model.supervisor is None:
                print("Phase supervisor: skipped, the ablation has no supervisor")
                self.phase_index, self.step = self.phase_index + 1, 0
                continue
            self._run_phase(phase, steps[phase], step_fns[phase])
            self.phase_index, self.step = self.phase_index + 1, 0
            self.save()

    def _run_phase(self, phase: str, total: int, step_fn) -> None:
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
# Plots and the final test evaluation
# ---------------------------------------------------------------------------

def plot_history(history: dict, path: Path, kl_target: float = 0.1) -> None:
    """Save one figure with the curves of every phase.

    Each phase keeps its own step counter, so each panel has its own x-axis.
    Validation KL panels show the spec target of 0.1 as a dashed line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(kind: str, phase: str, key: str):
        rows = [r for r in history[kind] if r["phase"] == phase and key in r]
        return [r["step"] for r in rows], [r[key] for r in rows]

    panels = [
        ("Phase 1: reconstruction MSE", "autoencoder", [("train", "recon_mse", "train"), ("val", "val_recon_mse", "validation")], True),
        ("Phase 2: supervised MSE", "supervisor", [("train", "supervised_mse", "train"), ("val", "val_supervised_mse", "validation")], True),
        ("Phase 3: generator terms", "joint", [("train", "g_adv", "adversarial"), ("train", "g_adv_e", "adversarial, raw path"),
                                               ("train", "g_moment", "moment"), ("train", "g_supervised_mse", "supervised MSE")], True),
        ("Phase 3: discriminator", "joint", [("train", "d_real", "real BCE"), ("train", "d_fake", "fake BCE"),
                                             ("train", "d_updated", "share of steps updated")], False),
        ("Phase 3: gradient norms", "joint", [("train", "g_grad_norm", "generator"), ("train", "d_grad_norm", "discriminator")], True),
        ("Phase 3: autoencoder on validation", "joint", [("train", "e_recon_mse", "train"), ("val", "val_recon_mse", "validation")], True),
        ("Phase 3: KL to validation hour", "joint", [("val", "val_kl_spread", "spread"), ("val", "val_kl_return", "return")], True),
        ("Phase 3: invalid books", "joint", [("val", "crossed_rate", "crossed or locked"), ("val", "ladder_rate", "ladder"),
                                             ("val", "negative_size_rate", "negative size")], False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, (title, phase, lines, log_y) in zip(axes.flat, panels):
        drawn, plotted_values = False, []
        for kind, key, label in lines:
            x, y = series(kind, phase, key)
            if x:
                ax.plot(x, y, marker="o" if kind == "val" else None, markersize=3, label=label)
                plotted_values += y
                drawn = True
        if "KL" in title:
            ax.axhline(kl_target, color="grey", linestyle="--", linewidth=1, label="target 0.1")
        if drawn and log_y and all(v > 0 for v in plotted_values):   # log axis only when every value is positive
            ax.set_yscale("log")
        ax.set_title(title if drawn else f"{title} (not run)")
        ax.set_xlabel("step in phase")
        if drawn:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def evaluate_test(run_dir: Path, splits: LOBSplits, args: argparse.Namespace, n_samples: int = 1024) -> dict:
    """Score the best checkpoint on the held-out test hour (15:00 to 15:50).

    Reports reconstruction error, invalid-book rates, KL of spreads and moves
    against the test hour and against the training reference, the same KL for
    real data alone (the floor any generator is judged against), and the cost
    of sampling.
    """
    checkpoint = run_dir / "best.pt" if (run_dir / "best.pt").exists() else run_dir / "last.pt"
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model = model_from_state(state, args.device)

    evaluator = Evaluator(splits, args.representation, args.device, n_samples, args.seed)
    test_x, test_anchor = Evaluator._stack(splits.test, range(len(splits.test)))
    move_flag = splits.config.get("move_flag", False)
    test_books = decode_windows(test_x, splits.scaler, test_anchor, args.representation, move_flag)

    torch.manual_seed(args.seed + 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    fake_x = model.sample(n_samples, splits.test.seq_len, args.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sample_seconds = time.time() - start
    pick = torch.randint(len(test_anchor), (n_samples,), generator=torch.Generator().manual_seed(args.seed))
    fake = decode_windows(fake_x, splits.scaler, test_anchor[pick], args.representation, move_flag)

    x = test_x.to(args.device)
    report = {
        "checkpoint": checkpoint.name,
        "checkpoint_phase_step": [PHASES[min(state["phase_index"], 2)], state["step"]],
        "test_recon_mse": float(torch.mean((model.reconstruct(x) - x) ** 2)) if isinstance(model, TimeGAN) else None,
        "violations": book_violations(fake),
        "vs_test_hour": distribution_report(test_books, fake),
        "vs_training_reference": distribution_report(evaluator.train_books, fake),
        "real_floor_training_vs_test": distribution_report(test_books, evaluator.train_books),
        "real_floor_validation_vs_test": distribution_report(test_books, evaluator.val_books),
        "parameters": model.parameter_counts(),
        "sampling": {"sequences": n_samples, "seconds": sample_seconds,
                     "sequences_per_second": n_samples / max(sample_seconds, 1e-9)},
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(report, indent=1))

    print(f"\nTest hour evaluation of {checkpoint.name} ({n_samples} synthetic windows vs {len(test_books)} real)")
    print(f"  invalid books: {report['violations']}")
    row = "  {:32s} spread KL {:7.3f}   return KL {:7.3f}"
    for label, key in [("synthetic vs test hour", "vs_test_hour"), ("synthetic vs training reference", "vs_training_reference"),
                       ("real floor: training vs test", "real_floor_training_vs_test"),
                       ("real floor: validation vs test", "real_floor_validation_vs_test")]:
        print(row.format(label, report[key]["kl_spread"], report[key]["kl_return"]))
    print(f"  sampling: {report['sampling']['sequences_per_second']:.0f} sequences per second")
    return report


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    run_dir = Path(args.out_dir) / args.run_name
    if args.resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)["args"]
        saved.update(resume=True, device=args.device)
        args = argparse.Namespace(**saved)
    elif (run_dir / "last.pt").exists():
        raise FileExistsError(f"{run_dir} already has a run; pass --resume or choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    splits = build_datasets(args.data_dir, seq_len=args.seq_len, representation=args.representation,
                            event_stride=args.event_stride, scaling=args.scaling,
                            move_flag=getattr(args, "move_flag", False))
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
    evaluate_test(run_dir, splits, args)
    return run_dir


if __name__ == "__main__":
    main()
