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

import torch
import torch.nn as nn

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
