"""TimeGAN networks and training objectives for limit order book sequences.

Implements the model of

    J. Yoon, D. Jarrett and M. van der Schaar, "Time-series Generative
    Adversarial Networks", Advances in Neural Information Processing
    Systems 32 (NeurIPS 2019).

Notation used in every docstring below: B batch size, T sequence length,
F number of features, H latent (hidden) size, Z noise size.

Data flow:
    real       x (B, T, F) --embedder--> h (B, T, H) --recovery--> x_tilde (B, T, F)
    synthetic  z (B, T, Z) --generator--> e_hat (B, T, H) --supervisor--> h_hat (B, T, H)
               --recovery--> x_hat (B, T, F)
    the discriminator maps any latent sequence (B, T, H) to per-step logits (B, T, 1).

The adversarial game is played on latent sequences, not on raw features, and
the supervisor adds a step-ahead regression loss in that latent space.

This file uses PyTorch only (no NumPy, no pandas), as the spec requires.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RNN_TYPES = {"gru": nn.GRU, "lstm": nn.LSTM}
ACTIVATIONS = ("sigmoid", "identity")


# ---------------------------------------------------------------------------
# Building block shared by all five networks
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    """A stacked GRU (or LSTM) followed by a linear head applied at every step.

    Input (B, T, in_dim) -> output (B, T, out_dim). The head reads the top
    layer's hidden state at each step, so with a unidirectional RNN output
    step t depends only on input steps 1..t. That causality is what lets the
    supervisor's output at step t act as a prediction of step t + 1.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        rnn_type: str = "gru",
        activation: str = "sigmoid",
        bidirectional: bool = False,
    ):
        super().__init__()
        if rnn_type not in RNN_TYPES:
            raise ValueError(f"rnn_type must be one of {tuple(RNN_TYPES)}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.rnn = RNN_TYPES[rnn_type](
            in_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_dim * directions, out_dim)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier input and head weights, orthogonal recurrent weights, zero biases.

        PyTorch stores the gate matrices of one layer stacked along dim 0
        (3 for a GRU, 4 for an LSTM), so each gate's square recurrent matrix
        is made orthogonal separately. An orthogonal recurrent matrix neither
        shrinks nor grows the hidden state at initialisation, which helps
        gradients survive backpropagation through 64 steps (Saxe et al., 2014).
        """
        for name, p in self.rnn.named_parameters():
            if name.startswith("weight_hh"):
                for gate in p.data.chunk(p.shape[0] // p.shape[1], dim=0):
                    nn.init.orthogonal_(gate)
            elif name.startswith("weight_ih"):
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)        # (B, T, hidden_dim * directions), top layer at every step
        y = self.head(out)          # (B, T, out_dim)
        return torch.sigmoid(y) if self.activation == "sigmoid" else y


# ---------------------------------------------------------------------------
# The five TimeGAN networks
# ---------------------------------------------------------------------------

class Embedder(RecurrentBlock):
    """e: real features x (B, T, F) -> latent codes h (B, T, H).

    The sigmoid output keeps every latent code in (0, 1), as in the paper, so
    the generator (also sigmoid) produces codes in the same box.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru"):
        super().__init__(feature_dim, hidden_dim, hidden_dim, num_layers, rnn_type, "sigmoid")


class Recovery(RecurrentBlock):
    """r: latent codes h (B, T, H) -> reconstructed features x_tilde (B, T, F).

    activation="sigmoid" suits min-max scaled data in [0, 1]; use "identity"
    for z-scored ("standard") data, which is unbounded.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru",
                 activation: str = "sigmoid"):
        super().__init__(hidden_dim, hidden_dim, feature_dim, num_layers, rnn_type, activation)


class Generator(RecurrentBlock):
    """g: noise z (B, T, Z) -> raw synthetic latent sequence e_hat (B, T, H) in (0, 1).

    One noise vector per step, so the generator can inject fresh randomness
    at every step while its recurrent state carries the temporal context.
    """

    def __init__(self, z_dim: int, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru"):
        super().__init__(z_dim, hidden_dim, hidden_dim, num_layers, rnn_type, "sigmoid")


class Supervisor(RecurrentBlock):
    """s: latent sequence (B, T, H) -> next-step latent predictions (B, T, H).

    Output step t is read as the prediction of h_{t+1} from h_1..h_t. It uses
    one layer fewer than the other networks, following the authors' code
    (never fewer than one layer).
    """

    def __init__(self, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru"):
        super().__init__(hidden_dim, hidden_dim, hidden_dim, max(1, num_layers - 1), rnn_type, "sigmoid")


class Discriminator(RecurrentBlock):
    """d: latent sequence (B, T, H) -> per-step real/fake logits (B, T, 1).

    No sigmoid at the output: the losses use binary cross-entropy with logits,
    which is numerically stable when the discriminator is very confident.
    The paper describes a bidirectional discriminator while the authors' code
    uses a unidirectional one; the default follows the code.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru", bidirectional: bool = False):
        super().__init__(hidden_dim, hidden_dim, 1, num_layers, rnn_type, "identity", bidirectional)


def count_parameters(module: nn.Module | None) -> int:
    """Total number of parameters in a module (0 for None)."""
    return 0 if module is None else sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

@contextmanager
def frozen(*modules: nn.Module | None):
    """Temporarily stop parameter gradients for the given modules.

    Gradients still flow *through* a frozen module to its inputs, only its own
    weights get none. Autograd decides this when the forward pass runs, so the
    flags can be restored before backward() is called. This keeps each loss
    from leaving stray gradients on networks that its phase does not train.
    """
    params = [p for m in modules if m is not None for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


def adversarial_bce(logits: torch.Tensor, real: bool) -> torch.Tensor:
    """Binary cross-entropy of per-step logits against an all-real or all-fake label."""
    target = torch.ones_like(logits) if real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Two-moment matching loss L_V between a synthetic and a real batch.

    Means and standard deviations are taken over the batch dimension, giving
    one value per (time step, feature) as in the authors' code, then the mean
    absolute differences are added. Population variance (correction=0); the
    1e-6 inside the square root keeps the gradient finite when a feature has
    zero variance in the batch.

    x_hat, x: (B, T, F) in the same (scaled) feature space. Returns a scalar.
    """
    std_hat = torch.sqrt(x_hat.var(dim=0, correction=0) + 1e-6)
    std_real = torch.sqrt(x.var(dim=0, correction=0) + 1e-6)
    std_term = (std_hat - std_real).abs().mean()
    mean_term = (x_hat.mean(dim=0) - x.mean(dim=0)).abs().mean()
    return std_term + mean_term


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------

@dataclass
class TimeGANConfig:
    """Architecture and loss weights for TimeGAN.

    Architecture. hidden_dim is both the GRU width and the latent size H.
    The paper uses 3 layers and a small hidden size (24 for 6 stock features);
    64 here keeps the latent at least as wide as the 40 order book features so
    the autoencoder is not forced to compress. z_dim defaults to feature_dim,
    as in the paper.

    Loss weights. The defaults follow the authors' released code, not the
    numbers written in the paper; check both before relying on them.
      code  (sqrt_losses=True):  embedder  recon_weight * sqrt(L_R) + embed_supervised_weight * L_S
                                            = 10 * sqrt(L_R) + 0.1 * L_S
                                 generator L_U + gamma * L_U_e + eta * sqrt(L_S) + moment_weight * L_V
                                            with eta = 100, moment_weight = 100, gamma = 1
      paper (sqrt_losses=False, recon_weight=1, embed_supervised_weight=1, eta=10):
                                 embedder  L_R + lambda * L_S with lambda = 1
                                 generator L_U + eta * L_S with eta = 10 (plus L_V from the code)
    Whichever convention is used must stay fixed across the model and its ablation.

    use_supervisor=False is the "no supervised loss" ablation: there is no
    supervisor network, the generator's output is used directly as the
    synthetic latent sequence (the supervisor is replaced by the identity),
    every L_S term is dropped, and phase 2 is skipped. All other terms and
    weights are unchanged, so any difference in the results is attributable
    to the supervised loss.

    d_threshold: the authors' code skips the discriminator update whenever its
    loss is already below 0.15, so it cannot overpower the generator.
    """

    feature_dim: int
    hidden_dim: int = 64
    num_layers: int = 3
    rnn_type: str = "gru"
    z_dim: int | None = None
    recovery_activation: str = "sigmoid"
    bidirectional_discriminator: bool = False
    use_supervisor: bool = True
    gamma: float = 1.0
    eta: float = 100.0
    moment_weight: float = 100.0
    recon_weight: float = 10.0
    embed_supervised_weight: float = 0.1
    sqrt_losses: bool = True
    d_threshold: float = 0.15

    def __post_init__(self):
        if self.z_dim is None:
            self.z_dim = self.feature_dim
        if self.recovery_activation not in ACTIVATIONS:
            raise ValueError(f"recovery_activation must be one of {ACTIVATIONS}")
        if self.rnn_type not in RNN_TYPES:
            raise ValueError(f"rnn_type must be one of {tuple(RNN_TYPES)}")


class TimeGAN(nn.Module):
    """The five TimeGAN networks plus one loss method per training phase.

    Training phases (train.py drives them; each loss returns (loss, logs),
    where loss is the scalar to backpropagate and logs holds detached scalar
    components for logging):
      1. autoencoder   autoencoder_loss(x)                     -> embedder, recovery
      2. supervisor    supervisor_loss(x)                      -> supervisor
      3. joint         generator_loss(x)                       -> generator, supervisor
                       autoencoder_loss(x, with_supervised=True) -> embedder, recovery
                       discriminator_loss(x)                   -> discriminator
    Each loss freezes or detaches every network outside its target group, so
    backward() only fills gradients for the networks that phase updates. The
    matching parameter lists for the optimisers come from *_parameters().
    """

    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = c = config
        self.embedder = Embedder(c.feature_dim, c.hidden_dim, c.num_layers, c.rnn_type)
        self.recovery = Recovery(c.feature_dim, c.hidden_dim, c.num_layers, c.rnn_type, c.recovery_activation)
        self.generator = Generator(c.z_dim, c.hidden_dim, c.num_layers, c.rnn_type)
        self.supervisor = Supervisor(c.hidden_dim, c.num_layers, c.rnn_type) if c.use_supervisor else None
        self.discriminator = Discriminator(c.hidden_dim, c.num_layers, c.rnn_type, c.bidirectional_discriminator)

    # -- bookkeeping ---------------------------------------------------------

    def networks(self) -> dict[str, nn.Module]:
        """The networks that exist in this model, by name (no supervisor in the ablation)."""
        nets = {"embedder": self.embedder, "recovery": self.recovery, "generator": self.generator,
                "supervisor": self.supervisor, "discriminator": self.discriminator}
        return {name: net for name, net in nets.items() if net is not None}

    def parameter_counts(self) -> dict[str, int]:
        """Parameters per network and in total, for the README resource table."""
        counts = {name: count_parameters(net) for name, net in self.networks().items()}
        counts["total"] = count_parameters(self)
        return counts

    def autoencoder_parameters(self) -> list[nn.Parameter]:
        return [*self.embedder.parameters(), *self.recovery.parameters()]

    def supervisor_parameters(self) -> list[nn.Parameter]:
        return [] if self.supervisor is None else list(self.supervisor.parameters())

    def generator_parameters(self) -> list[nn.Parameter]:
        """Generator plus supervisor: both shape the synthetic latent path in phase 3."""
        return [*self.generator.parameters(), *self.supervisor_parameters()]

    def discriminator_parameters(self) -> list[nn.Parameter]:
        return list(self.discriminator.parameters())

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    # -- forward maps ----------------------------------------------------------

    def noise(self, n: int, seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Noise z of shape (n, seq_len, Z), i.i.d. uniform on [0, 1) as in the paper."""
        return torch.rand(n, seq_len, self.config.z_dim, device=device or self._device())

    def supervise(self, e: torch.Tensor) -> torch.Tensor:
        """s(e) of shape (B, T, H); the identity in the ablation, which has no supervisor."""
        return e if self.supervisor is None else self.supervisor(e)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """x_hat = r(s(g(z))): noise (B, T, Z) -> synthetic features (B, T, F), with gradients."""
        return self.recovery(self.supervise(self.generator(z)))

    forward = generate

    @torch.no_grad()
    def sample(self, n: int, seq_len: int, device: torch.device | str | None = None,
               batch_size: int = 1024) -> torch.Tensor:
        """Draw n synthetic sequences (n, seq_len, F) in the scaled feature space.

        The output is still scaled; invert it with the training scaler from
        dataset.py. Generated in chunks of batch_size to bound memory.
        """
        device = device or self._device()
        chunks = [self.generate(self.noise(min(batch_size, n - i), seq_len, device))
                  for i in range(0, n, batch_size)]
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """x_tilde = r(e(x)), shape (B, T, F), for checking the autoencoder on held-out data."""
        return self.recovery(self.embedder(x))

    # -- losses, one per training phase ---------------------------------------

    def _supervised_mse(self, h: torch.Tensor) -> torch.Tensor:
        """L_S: supervisor output at step t against the true latent code at step t + 1.

        h: (B, T, H) real latent codes. Teacher forcing: the supervisor always
        reads the real codes h_1..h_t, never its own predictions.
        """
        return F.mse_loss(self.supervisor(h)[:, :-1], h[:, 1:])

    def _root(self, loss: torch.Tensor) -> torch.Tensor:
        """sqrt(loss) under the code's convention, the loss itself under the paper's."""
        return loss.sqrt() if self.config.sqrt_losses else loss

    def autoencoder_loss(self, x: torch.Tensor, with_supervised: bool = False):
        """Embedder and recovery objective (phase 1, and their half of phase 3).

        x: (B, T, F) real scaled windows.
        L_R = MSE(x, r(e(x))); the loss is recon_weight * sqrt(L_R) under the
        code convention (10 * sqrt(L_R)). With with_supervised=True (phase 3)
        it adds embed_supervised_weight * L_S (0.1 * L_S), which pushes the
        embedder towards a latent space the supervisor can predict; the
        supervisor itself is frozen here and trained by generator_loss. The
        ablation has no supervisor, so the flag has no effect there.
        Gradients reach the embedder and recovery only.
        Logs: recon_mse (L_R), supervised_mse (L_S, phase 3 only), total.
        """
        h = self.embedder(x)
        recon = F.mse_loss(self.recovery(h), x)
        loss = self.config.recon_weight * self._root(recon)
        logs = {"recon_mse": recon.detach()}
        if with_supervised and self.supervisor is not None:
            with frozen(self.supervisor):
                supervised = self._supervised_mse(h)
            loss = loss + self.config.embed_supervised_weight * supervised
            logs["supervised_mse"] = supervised.detach()
        logs["total"] = loss.detach()
        return loss, logs

    def supervisor_loss(self, x: torch.Tensor):
        """Phase 2 objective L_S on real latent codes, training the supervisor only.

        x: (B, T, F). The embedder runs without gradients, so it is fixed in
        this phase. The authors' code puts the generator in this optimiser as
        well, but L_S does not depend on the generator, so it gets no update.
        Logs: supervised_mse.
        """
        if self.supervisor is None:
            raise RuntimeError("The no-supervisor ablation has no phase 2; skip it")
        with torch.no_grad():
            h = self.embedder(x)
        loss = self._supervised_mse(h)
        return loss, {"supervised_mse": loss.detach(), "total": loss.detach()}

    def generator_loss(self, x: torch.Tensor, z: torch.Tensor | None = None):
        """Generator and supervisor objective in phase 3.

        x: (B, T, F) real windows; z: (B, T, Z) noise, drawn if not given.
        e_hat = g(z) is the raw synthetic latent path and h_hat = s(e_hat) the
        supervised one; x_hat = r(h_hat) its decoded features.
          L_U    BCE(d(h_hat), real)   non-saturating adversarial term
          L_U_e  BCE(d(e_hat), real)   same on the raw path, weight gamma
          L_S    step-ahead MSE on real codes, weight eta (on sqrt(L_S) under the code convention)
          L_V    moment_loss(x_hat, x), weight moment_weight
        The recovery and discriminator are frozen (gradients pass through them
        to the generator but do not change them); the real codes h are
        computed without gradients. Gradients reach the generator and the
        supervisor only. In the ablation h_hat equals e_hat and L_S is absent.
        Logs: adv, adv_e, supervised_mse (if any), moment, total.
        """
        c = self.config
        if z is None:
            z = self.noise(x.shape[0], x.shape[1], x.device)
        with torch.no_grad():
            h = self.embedder(x)
        with frozen(self.recovery, self.discriminator):
            e_hat = self.generator(z)
            h_hat = self.supervise(e_hat)
            x_hat = self.recovery(h_hat)
            adv = adversarial_bce(self.discriminator(h_hat), real=True)
            adv_e = adversarial_bce(self.discriminator(e_hat), real=True)
        moments = moment_loss(x_hat, x)
        loss = adv + c.gamma * adv_e + c.moment_weight * moments
        logs = {"adv": adv.detach(), "adv_e": adv_e.detach(), "moment": moments.detach()}
        if self.supervisor is not None:
            supervised = self._supervised_mse(h)
            loss = loss + c.eta * self._root(supervised)
            logs["supervised_mse"] = supervised.detach()
        logs["total"] = loss.detach()
        return loss, logs

    def discriminator_loss(self, x: torch.Tensor, z: torch.Tensor | None = None):
        """Discriminator objective in phase 3.

        x: (B, T, F) real windows; z: (B, T, Z) noise, drawn if not given.
        BCE(d(h), real) + BCE(d(h_hat), fake) + gamma * BCE(d(e_hat), fake),
        with every latent input computed without gradients, so only the
        discriminator is trained. In the ablation h_hat equals e_hat, so the
        fake term carries weight 1 + gamma, exactly what the full objective
        gives when the supervisor is the identity.
        Logs: real, fake, fake_e, total.
        """
        if z is None:
            z = self.noise(x.shape[0], x.shape[1], x.device)
        with torch.no_grad():
            h = self.embedder(x)
            e_hat = self.generator(z)
            h_hat = self.supervise(e_hat)
        real = adversarial_bce(self.discriminator(h), real=True)
        fake = adversarial_bce(self.discriminator(h_hat), real=False)
        fake_e = adversarial_bce(self.discriminator(e_hat), real=False)
        loss = real + fake + self.config.gamma * fake_e
        logs = {"real": real.detach(), "fake": fake.detach(), "fake_e": fake_e.detach(), "total": loss.detach()}
        return loss, logs

    def should_update_discriminator(self, d_loss: torch.Tensor | float) -> bool:
        """True when the discriminator loss is above d_threshold (0.15 in the authors' code).

        Usage in train.py: compute discriminator_loss on the batch, and only
        call backward() and step the optimiser when this returns True.
        """
        if isinstance(d_loss, torch.Tensor):
            d_loss = d_loss.detach()   # reading the value must not touch the autograd graph
        return float(d_loss) > self.config.d_threshold


# ---------------------------------------------------------------------------
# Smoke test: python modules.py [--data-dir ../../../../data/LOBSTER]
# ---------------------------------------------------------------------------

def _load_batch(data_dir: str | None, batch_size: int, seq_len: int, feature_dim: int, seed: int) -> torch.Tensor:
    """A random batch of real training windows, or uniform noise when no data folder is given."""
    g = torch.Generator().manual_seed(seed)
    if data_dir is None:
        print(f"No --data-dir given, using random data of shape ({batch_size}, {seq_len}, {feature_dim})")
        return torch.rand(batch_size, seq_len, feature_dim, generator=g)
    from dataset import build_datasets  # imported here so modules.py itself never needs pandas
    splits = build_datasets(data_dir, seq_len=seq_len, representation="structured")
    idx = torch.randperm(len(splits.train), generator=g)[:batch_size]
    print(f"Loaded {len(idx)} of {len(splits.train)} real training windows from {data_dir}")
    return torch.stack([splits.train[int(i)] for i in idx])


def _check_gradients(model: TimeGAN, trained: set[str], phase: str) -> None:
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


def _check_model(x: torch.Tensor, config: TimeGANConfig, label: str) -> None:
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
        _check_gradients(model, trained & set(model.networks()), phase)
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


def _overfit_autoencoder(x: torch.Tensor, config: TimeGANConfig, steps: int) -> None:
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


if __name__ == "__main__":
    import argparse

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
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"torch {torch.__version__}, device {dev}"
          + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""))
    batch = _load_batch(args.data_dir, args.batch_size, args.seq_len, args.feature_dim, seed=0).to(dev)

    base = dict(feature_dim=batch.shape[-1], hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                rnn_type=args.rnn_type)
    _check_model(batch, TimeGANConfig(**base), "TimeGAN")
    _check_model(batch, TimeGANConfig(**base, use_supervisor=False), "ablation: no supervisor")
    if args.overfit_steps > 0:
        _overfit_autoencoder(batch, TimeGANConfig(**base), args.overfit_steps)
    print("\nAll checks passed.")
