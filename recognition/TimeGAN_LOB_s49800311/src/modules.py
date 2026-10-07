"""TimeGAN and a baseline recurrent GAN for order book sequences.

TimeGAN follows Yoon, Jarrett and van der Schaar, "Time-series Generative
Adversarial Networks", NeurIPS 2019. Shapes use B batch, T sequence length,
F features, H latent size and Z noise size.

    real       x (B, T, F) -> embedder -> h (B, T, H) -> recovery -> x_tilde
    synthetic  z (B, T, Z) -> generator -> e_hat -> supervisor -> h_hat -> recovery -> x_hat
    the discriminator scores latent sequences (B, T, H) with one logit per step

PyTorch only, as the spec requires for this file.
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
# Networks
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    """Stacked GRU or LSTM with a linear head at every step: (B, T, in) -> (B, T, out).

    With a unidirectional RNN, output step t only sees inputs 1..t, which is
    what lets the supervisor's output at t act as a prediction of t + 1.
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
        self.rnn = RNN_TYPES[rnn_type](in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                       bidirectional=bidirectional)
        directions = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_dim * directions, out_dim)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier input and head weights, orthogonal recurrent weights, zero biases.

        PyTorch stacks the gate matrices of a layer along dim 0, so each gate is
        made orthogonal on its own. Orthogonal recurrent weights help gradients
        survive 64 steps of backpropagation (Saxe et al., 2014).
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
        out, _ = self.rnn(x)
        y = self.head(out)
        return torch.sigmoid(y) if self.activation == "sigmoid" else y


class Embedder(RecurrentBlock):
    """Features x (B, T, F) -> latent codes h (B, T, H) in (0, 1)."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru"):
        super().__init__(feature_dim, hidden_dim, hidden_dim, num_layers, rnn_type, "sigmoid")


class Recovery(RecurrentBlock):
    """Latent codes h (B, T, H) -> features (B, T, F).

    Sigmoid output for min-max scaled data, identity for standard scaling.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru",
                 activation: str = "sigmoid"):
        super().__init__(hidden_dim, hidden_dim, feature_dim, num_layers, rnn_type, activation)


class Generator(RecurrentBlock):
    """Noise z (B, T, Z) -> synthetic latent sequence e_hat (B, T, H) in (0, 1)."""

    def __init__(self, z_dim: int, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru"):
        super().__init__(z_dim, hidden_dim, hidden_dim, num_layers, rnn_type, "sigmoid")


class Supervisor(RecurrentBlock):
    """Latent sequence (B, T, H) -> prediction of the next latent step (B, T, H).

    One layer fewer than the other networks, as in the authors' code.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru"):
        super().__init__(hidden_dim, hidden_dim, hidden_dim, max(1, num_layers - 1), rnn_type, "sigmoid")


class Discriminator(RecurrentBlock):
    """Latent sequence (B, T, H) -> real/fake logits (B, T, 1).

    Outputs logits, not probabilities, because the losses use BCE with logits.
    Unidirectional by default like the authors' code (the paper says bidirectional).
    """

    def __init__(self, hidden_dim: int, num_layers: int = 3, rnn_type: str = "gru", bidirectional: bool = False):
        super().__init__(hidden_dim, hidden_dim, 1, num_layers, rnn_type, "identity", bidirectional)


def count_parameters(module: nn.Module | None) -> int:
    return 0 if module is None else sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

@contextmanager
def frozen(*modules: nn.Module | None):
    """Stop the given modules' weights from receiving gradients inside the block.

    Gradients still flow through them to their inputs. Autograd records this
    during the forward pass, so the flags can be restored before backward().
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
    """BCE of per-step logits against an all-real or all-fake target."""
    target = torch.ones_like(logits) if real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """L_V: mean absolute gap in batch mean and batch std, per (step, feature).

    The 1e-6 keeps the square root's gradient finite for a constant feature.
    """
    std_hat = torch.sqrt(x_hat.var(dim=0, correction=0) + 1e-6)
    std_real = torch.sqrt(x.var(dim=0, correction=0) + 1e-6)
    std_term = (std_hat - std_real).abs().mean()
    mean_term = (x_hat.mean(dim=0) - x.mean(dim=0)).abs().mean()
    return std_term + mean_term


# ---------------------------------------------------------------------------
# Shared interface
# ---------------------------------------------------------------------------

class SequenceGAN(nn.Module):
    """What train.py and predict.py need from any model: noise, sampling, parameter counts.

    Subclasses provide networks(), generate(z), generator_loss,
    discriminator_loss and should_update_discriminator, and keep z_dim in self.config.
    """

    def networks(self) -> dict[str, nn.Module]:
        raise NotImplementedError

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def parameter_counts(self) -> dict[str, int]:
        counts = {name: count_parameters(net) for name, net in self.networks().items()}
        counts["total"] = count_parameters(self)
        return counts

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def noise(self, n: int, seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Uniform noise of shape (n, seq_len, Z + S).

        The first Z channels are new at every step. The last S channels
        (static_noise_dim, TimeGAN only) are drawn once per sequence and
        repeated, so a whole window can share one regime, such as a wide spread.
        """
        device = device or self._device()
        z = torch.rand(n, seq_len, self.config.z_dim, device=device)
        static_dim = getattr(self.config, "static_noise_dim", 0)     # the baseline config has none
        if static_dim == 0:
            return z
        static = torch.rand(n, 1, static_dim, device=device).expand(n, seq_len, static_dim)
        return torch.cat([z, static], dim=2)

    @torch.no_grad()
    def sample(self, n: int, seq_len: int, device: torch.device | str | None = None,
               batch_size: int = 1024) -> torch.Tensor:
        """n synthetic sequences (n, seq_len, F), still in the scaled feature space."""
        device = device or self._device()
        chunks = [self.generate(self.noise(min(batch_size, n - i), seq_len, device))
                  for i in range(0, n, batch_size)]
        return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# TimeGAN
# ---------------------------------------------------------------------------

@dataclass
class TimeGANConfig:
    """Architecture and loss weights for TimeGAN.

    hidden_dim is both the GRU width and the latent size. 64 keeps the latent
    at least as wide as the 40 features; z_dim defaults to feature_dim.

    The default weights follow the authors' code rather than the paper:
        embedder   10 * sqrt(L_R) + 0.1 * L_S
        generator  L_U + gamma * L_U_e + 100 * sqrt(L_S) + 100 * L_V
    The paper's version is sqrt_losses=False, recon_weight=1,
    embed_supervised_weight=1 and eta=10.

    use_supervisor=False is the ablation: no supervisor (the identity takes
    its place), no L_S terms and no phase 2, with everything else unchanged.
    The discriminator is skipped when its loss is below d_threshold.
    static_noise_dim adds noise channels held constant per window (0 = paper).
    """

    feature_dim: int
    hidden_dim: int = 64
    num_layers: int = 3
    rnn_type: str = "gru"
    z_dim: int | None = None
    static_noise_dim: int = 0
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


class TimeGAN(SequenceGAN):
    """The five TimeGAN networks and one loss per training phase.

        phase 1  autoencoder_loss(x)                       trains embedder, recovery
        phase 2  supervisor_loss(x)                        trains supervisor
        phase 3  generator_loss(x)                         trains generator, supervisor
                 autoencoder_loss(x, with_supervised=True) trains embedder, recovery
                 discriminator_loss(x)                     trains discriminator

    Every loss returns (loss, logs) and only leaves gradients on the networks
    it trains. logs holds detached values for the training history.
    """

    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = c = config
        self.embedder = Embedder(c.feature_dim, c.hidden_dim, c.num_layers, c.rnn_type)
        self.recovery = Recovery(c.feature_dim, c.hidden_dim, c.num_layers, c.rnn_type, c.recovery_activation)
        self.generator = Generator(c.z_dim + c.static_noise_dim, c.hidden_dim, c.num_layers, c.rnn_type)
        self.supervisor = Supervisor(c.hidden_dim, c.num_layers, c.rnn_type) if c.use_supervisor else None
        self.discriminator = Discriminator(c.hidden_dim, c.num_layers, c.rnn_type, c.bidirectional_discriminator)

    def networks(self) -> dict[str, nn.Module]:
        nets = {"embedder": self.embedder, "recovery": self.recovery, "generator": self.generator,
                "supervisor": self.supervisor, "discriminator": self.discriminator}
        return {name: net for name, net in nets.items() if net is not None}

    def autoencoder_parameters(self) -> list[nn.Parameter]:
        return [*self.embedder.parameters(), *self.recovery.parameters()]

    def supervisor_parameters(self) -> list[nn.Parameter]:
        return [] if self.supervisor is None else list(self.supervisor.parameters())

    def generator_parameters(self) -> list[nn.Parameter]:
        """Generator and supervisor, which are trained together in phase 3."""
        return [*self.generator.parameters(), *self.supervisor_parameters()]

    def discriminator_parameters(self) -> list[nn.Parameter]:
        return list(self.discriminator.parameters())

    def supervise(self, e: torch.Tensor) -> torch.Tensor:
        """The supervisor's output, or e itself in the ablation."""
        return e if self.supervisor is None else self.supervisor(e)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """x_hat = recovery(supervisor(generator(z)))."""
        return self.recovery(self.supervise(self.generator(z)))

    forward = generate

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.recovery(self.embedder(x))

    def _supervised_mse(self, h: torch.Tensor) -> torch.Tensor:
        """L_S: the supervisor's output at step t against the real code at t + 1."""
        return F.mse_loss(self.supervisor(h)[:, :-1], h[:, 1:])

    def _root(self, loss: torch.Tensor) -> torch.Tensor:
        """sqrt(loss) under the code's convention, loss under the paper's."""
        return loss.sqrt() if self.config.sqrt_losses else loss

    def autoencoder_loss(self, x: torch.Tensor, with_supervised: bool = False):
        """recon_weight * sqrt(L_R), plus embed_supervised_weight * L_S in phase 3.

        The L_S term pushes the embedder towards codes the supervisor can
        predict; the supervisor itself is frozen here.
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
        """Phase 2: L_S on real latent codes, with the embedder fixed."""
        if self.supervisor is None:
            raise RuntimeError("The no-supervisor ablation has no phase 2")
        with torch.no_grad():
            h = self.embedder(x)
        loss = self._supervised_mse(h)
        return loss, {"supervised_mse": loss.detach(), "total": loss.detach()}

    def generator_loss(self, x: torch.Tensor, z: torch.Tensor | None = None):
        """Phase 3 loss for the generator and supervisor.

            L_U    adversarial loss on h_hat = supervisor(generator(z))
            L_U_e  adversarial loss on e_hat = generator(z), weight gamma
            L_S    supervised loss on real codes, weight eta
            L_V    moment loss on decoded features, weight moment_weight

        The recovery and discriminator are frozen, so gradients pass through
        them without changing them.
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
        """BCE on real codes, on h_hat and (weight gamma) on e_hat.

        Every input is computed without gradients, so only the discriminator
        learns. In the ablation h_hat equals e_hat.
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
        """Skip the discriminator step while its loss is below d_threshold."""
        if isinstance(d_loss, torch.Tensor):
            d_loss = d_loss.detach()
        return float(d_loss) > self.config.d_threshold


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

@dataclass
class RecurrentGANConfig:
    """Settings for the baseline recurrent GAN.

    A GRU generator writes features straight from noise and a GRU
    discriminator judges them, in the style of C-RNN-GAN (Mogren, 2016) and
    RCGAN (Esteban et al., 2017). There is no latent space and no supervisor.
    moment_weight > 0 adds TimeGAN's moment loss; at 0 it is only logged.
    """

    feature_dim: int
    hidden_dim: int = 64
    num_layers: int = 3
    rnn_type: str = "gru"
    z_dim: int | None = None
    output_activation: str = "sigmoid"
    moment_weight: float = 0.0

    def __post_init__(self):
        if self.z_dim is None:
            self.z_dim = self.feature_dim
        if self.output_activation not in ACTIVATIONS:
            raise ValueError(f"output_activation must be one of {ACTIVATIONS}")
        if self.rnn_type not in RNN_TYPES:
            raise ValueError(f"rnn_type must be one of {tuple(RNN_TYPES)}")


class RecurrentGAN(SequenceGAN):
    """Baseline: noise (B, T, Z) -> features (B, T, F), judged in feature space."""

    def __init__(self, config: RecurrentGANConfig):
        super().__init__()
        self.config = c = config
        self.generator = RecurrentBlock(c.z_dim, c.hidden_dim, c.feature_dim, c.num_layers, c.rnn_type,
                                        c.output_activation)
        self.discriminator = RecurrentBlock(c.feature_dim, c.hidden_dim, 1, c.num_layers, c.rnn_type, "identity")

    def networks(self) -> dict[str, nn.Module]:
        return {"generator": self.generator, "discriminator": self.discriminator}

    def generator_parameters(self) -> list[nn.Parameter]:
        return list(self.generator.parameters())

    def discriminator_parameters(self) -> list[nn.Parameter]:
        return list(self.discriminator.parameters())

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    forward = generate

    def generator_loss(self, x: torch.Tensor, z: torch.Tensor | None = None):
        """Adversarial loss with the discriminator frozen, plus the optional moment loss."""
        if z is None:
            z = self.noise(x.shape[0], x.shape[1], x.device)
        with frozen(self.discriminator):
            x_hat = self.generator(z)
            adv = adversarial_bce(self.discriminator(x_hat), real=True)
        moments = moment_loss(x_hat, x)
        loss = adv + self.config.moment_weight * moments if self.config.moment_weight > 0 else adv
        return loss, {"adv": adv.detach(), "moment": moments.detach(), "total": loss.detach()}

    def discriminator_loss(self, x: torch.Tensor, z: torch.Tensor | None = None):
        if z is None:
            z = self.noise(x.shape[0], x.shape[1], x.device)
        with torch.no_grad():
            x_hat = self.generator(z)
        real = adversarial_bce(self.discriminator(x), real=True)
        fake = adversarial_bce(self.discriminator(x_hat), real=False)
        loss = real + fake
        return loss, {"real": real.detach(), "fake": fake.detach(), "total": loss.detach()}

    def should_update_discriminator(self, d_loss: torch.Tensor | float) -> bool:
        return True
