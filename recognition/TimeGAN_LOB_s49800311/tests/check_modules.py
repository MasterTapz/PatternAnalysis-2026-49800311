"""Smoke test of the networks and losses in src/modules.py on one batch.

For TimeGAN, its no-supervisor ablation and the baseline recurrent GAN, every
loss runs forward and backward and exactly the intended networks must receive
finite, non-zero gradients (with every frozen flag restored afterwards);
sampling must give the right shape and range. A final test overfits the
autoencoder on one batch to show that phase 1 can learn more than the
per-feature means. Peak CUDA memory is reported for the README resource table.

Run from the project folder (the one holding src/ and tests/):
  python tests/check_modules.py --data-dir ../../../../data/LOBSTER
Without --data-dir a random batch is used instead of real windows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # make the src modules importable

from modules import RNN_TYPES, RecurrentGAN, RecurrentGANConfig, TimeGAN, TimeGANConfig  # noqa: E402


def load_batch(data_dir: str | None, batch_size: int, seq_len: int, feature_dim: int, seed: int) -> torch.Tensor:
    """A random batch of real training windows, or uniform noise when no data folder is given."""
    g = torch.Generator().manual_seed(seed)
    if data_dir is None:
        print(f"No --data-dir given, using random data of shape ({batch_size}, {seq_len}, {feature_dim})")
        return torch.rand(batch_size, seq_len, feature_dim, generator=g)
    from dataset import build_datasets  # imported here so the random-data path never needs pandas
    splits = build_datasets(data_dir, seq_len=seq_len, representation="structured")
    idx = torch.randperm(len(splits.train), generator=g)[:batch_size]
    print(f"Loaded {len(idx)} of {len(splits.train)} real training windows from {data_dir}")
    return torch.stack([splits.train[int(i)] for i in idx])


def check_gradients(model: TimeGAN, trained: set[str], phase: str) -> None:
    """Every parameter of the trained networks has a finite gradient; all others have none."""
    for name, net in model.networks().items():
        grads = [(pname, p.grad) for pname, p in net.named_parameters()]
        if name in trained:
            missing = [pname for pname, g in grads if g is None]
            assert not missing, f"{phase}: {name} parameters without gradient: {missing}"
            assert all(bool(torch.isfinite(g).all()) for _, g in grads), f"{phase}: non-finite gradient in {name}"
            assert any(bool(g.abs().sum() > 0) for _, g in grads), f"{phase}: all-zero gradient in {name}"
        else:
            stray = [pname for pname, g in grads if g is not None]
            assert not stray, f"{phase}: {name} should not be trained but got gradients on {stray}"
    assert all(p.requires_grad for p in model.parameters()), f"{phase}: a frozen flag was not restored"


def check_model(x: torch.Tensor, config: TimeGANConfig, label: str) -> None:
    """Run every phase loss forward and backward, check gradients, sample, report memory."""
    device = x.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = TimeGAN(config).to(device).train()
    print(f"\n[{label}] parameters: {model.parameter_counts()}")

    phases = [
        ("autoencoder", lambda: model.autoencoder_loss(x), {"embedder", "recovery"}),
        ("autoencoder+supervised", lambda: model.autoencoder_loss(x, with_supervised=True), {"embedder", "recovery"}),
        ("supervisor", lambda: model.supervisor_loss(x), {"supervisor"}),
        ("generator", lambda: model.generator_loss(x), {"generator", "supervisor"}),
        ("discriminator", lambda: model.discriminator_loss(x), {"discriminator"}),
    ]
    for phase, loss_fn, trained in phases:
        if phase == "supervisor" and not config.use_supervisor:
            try:
                loss_fn()
            except RuntimeError:
                print(f"  {phase:23s} skipped (ablation has no supervisor, as intended)")
                continue
            raise AssertionError("supervisor_loss should refuse to run in the ablation")
        model.zero_grad(set_to_none=True)
        loss, logs = loss_fn()
        assert loss.dim() == 0 and bool(torch.isfinite(loss)), f"{phase}: loss is not a finite scalar"
        assert all(bool(torch.isfinite(v)) for v in logs.values()), f"{phase}: non-finite log value"
        loss.backward()
        check_gradients(model, trained & set(model.networks()), phase)
        print(f"  {phase:23s} " + "  ".join(f"{k} {float(v):.4f}" for k, v in logs.items()))
        if phase == "discriminator":
            print(f"  {'':23s} update discriminator at this loss: {model.should_update_discriminator(loss)}")

    samples = model.sample(512, x.shape[1], device, batch_size=200)
    assert samples.shape == (512, x.shape[1], x.shape[2]), f"bad sample shape {tuple(samples.shape)}"
    assert bool(torch.isfinite(samples).all())
    if config.recovery_activation == "sigmoid":
        assert float(samples.min()) >= 0 and float(samples.max()) <= 1, "sigmoid output left [0, 1]"
    print(f"  sample                  shape {tuple(samples.shape)}  range [{float(samples.min()):.3f}, "
          f"{float(samples.max()):.3f}]")
    if device.type == "cuda":
        print(f"  peak CUDA memory        {torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB")


def check_baseline(x: torch.Tensor, config: RecurrentGANConfig) -> None:
    """Both baseline losses run, train only their own network, and sampling works."""
    model = RecurrentGAN(config).to(x.device).train()
    print(f"\n[baseline recurrent GAN] parameters: {model.parameter_counts()}")
    for phase, loss_fn, trained in [("generator", lambda: model.generator_loss(x), {"generator"}),
                                    ("discriminator", lambda: model.discriminator_loss(x), {"discriminator"})]:
        model.zero_grad(set_to_none=True)
        loss, logs = loss_fn()
        assert loss.dim() == 0 and bool(torch.isfinite(loss)), f"baseline {phase}: loss is not a finite scalar"
        loss.backward()
        for name, net in model.networks().items():
            has_grad = [p.grad is not None for p in net.parameters()]
            assert all(has_grad) if name in trained else not any(has_grad), f"baseline {phase}: wrong gradients on {name}"
        assert all(p.requires_grad for p in model.parameters()), f"baseline {phase}: a frozen flag was not restored"
        print(f"  {phase:23s} " + "  ".join(f"{k} {float(v):.4f}" for k, v in logs.items()))
    samples = model.sample(512, x.shape[1], x.device, batch_size=200)
    assert samples.shape == (512, x.shape[1], x.shape[2]) and bool(torch.isfinite(samples).all())
    print(f"  sample                  shape {tuple(samples.shape)}  range [{float(samples.min()):.3f}, "
          f"{float(samples.max()):.3f}]")


def overfit_autoencoder(x: torch.Tensor, config: TimeGANConfig, steps: int) -> None:
    """Phase 1 on a single batch: reconstruction must beat predicting each feature's mean.

    Falling from the initial error alone proves little: the recovery's output
    bias can learn the per-feature means within a few hundred steps and stop
    there. Outputting the batch mean of every feature gives an MSE equal to
    the average per-feature variance, so the check is against that baseline.
    """
    torch.manual_seed(0)
    model = TimeGAN(config).to(x.device).train()
    optimiser = torch.optim.Adam(model.autoencoder_parameters(), lr=1e-3)   # learning rate as in the paper
    mean_baseline = float(((x - x.mean(dim=(0, 1))) ** 2).mean())
    with torch.no_grad():
        before = float(model.autoencoder_loss(x)[1]["recon_mse"])
    print(f"\n[overfit] autoencoder on one batch, Adam lr 1e-3. Initial recon MSE {before:.5f}, "
          f"per-feature mean baseline {mean_baseline:.5f}")
    for step in range(1, steps + 1):
        optimiser.zero_grad(set_to_none=True)
        loss, logs = model.autoencoder_loss(x)
        loss.backward()
        optimiser.step()
        if step % 250 == 0 or step in (200, steps):
            print(f"  step {step:5d}  recon MSE {float(logs['recon_mse']):.5f}")
    with torch.no_grad():
        after = float(model.autoencoder_loss(x)[1]["recon_mse"])
    print(f"  final recon MSE {after:.5f}: {before / after:.1f}x below initial, "
          f"{mean_baseline / after:.2f}x below the mean baseline")
    assert after < 0.75 * mean_baseline, "autoencoder did not learn beyond the per-feature means"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke test the TimeGAN modules on one batch.")
    parser.add_argument("--data-dir", default=None, help="LOBSTER folder; random data is used if omitted")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=40, help="only used with random data")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--rnn-type", choices=tuple(RNN_TYPES), default="gru")
    parser.add_argument("--overfit-steps", type=int, default=1500,
                        help="the 3-layer autoencoder sits at the mean baseline for the first few hundred steps")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU even if CUDA is available")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"torch {torch.__version__}, device {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    batch = load_batch(args.data_dir, args.batch_size, args.seq_len, args.feature_dim, seed=0).to(device)

    base = dict(feature_dim=batch.shape[-1], hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                rnn_type=args.rnn_type)
    # The order matters for reproducibility: every model draws its initial weights from the global seed.
    check_model(batch, TimeGANConfig(**base), "TimeGAN")
    check_model(batch, TimeGANConfig(**base, use_supervisor=False), "ablation: no supervisor")
    check_baseline(batch, RecurrentGANConfig(**base))
    if args.overfit_steps > 0:
        overfit_autoencoder(batch, TimeGANConfig(**base), args.overfit_steps)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
