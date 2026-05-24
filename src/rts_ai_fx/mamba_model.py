"""
Mamba State-Space Model for Long-Sequence FX Forecasting.

Mamba (Gu & Dao, 2024) is a state-space model that achieves:
  - O(n) inference (vs O(n²) for Transformers)
  - 10,000+ timestep lookback (vs 30-100 for LSTM/TFT)
  - Better accuracy on long time series benchmarks

This is a simplified pure-PyTorch implementation that captures the
core Mamba architecture: selective state spaces with scan operation.

References:
  - Gu & Dao (2024), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  - Mamba-2: "Transformers are SSMs" (2024)
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Stub classes so module can be imported without torch
    class nn:  # type: ignore[no-redef]
        class Module:
            pass

        class Linear:
            pass

        class LayerNorm:
            pass

        class ModuleList:
            pass

        class Parameter:
            pass

        class Dropout:
            pass

        class SiLU:
            pass

        class Conv1d:
            pass

    class F:  # type: ignore[no-redef]
        @staticmethod
        def softplus(x):
            return x

        @staticmethod
        def sigmoid(x):
            return x


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Mamba selective scan operation (simplified).

    Computes hidden states h(t) where:
      h(t) = exp(delta[t] * A) * h(t-1) + delta[t] * B[t] * u[t]

    A is diagonal (d_state,), delta/B/u are (batch, seq, d_state).

    Returns:
        hidden states (batch, seq, d_state) — one state vector per timestep
    """
    batch, seq_len, d_state = B.shape[0], B.shape[1], A.shape[-1]
    delta = F.softplus(delta)
    # A is diagonal (d_state,) — broadcast over batch and seq
    discretized_A = torch.exp(
        delta * A.unsqueeze(0).unsqueeze(0)
    )  # (batch, seq, d_state)
    discretized_B = delta * B  # (batch, seq, d_state)

    h = torch.zeros(batch, d_state, device=u.device)
    hs = []
    for t in range(seq_len):
        h = discretized_A[:, t] * h + discretized_B[:, t] * u[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)  # (batch, seq, d_state)


class MambaBlock(nn.Module):
    """Single Mamba block with selective SSM + gated MLP.

    Architecture:
      x -> LayerNorm -> Conv1d -> SiLU -> SSM -> Gate -> residual
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection
        self.norm = nn.LayerNorm(d_model)
        self.conv1d = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
        )
        self.silu = nn.SiLU()

        # SSM parameters (A is diagonal for simplified selective scan)
        self.A = nn.Parameter(torch.randn(d_state) * 0.01)
        self.u_proj = nn.Linear(d_model, d_state)
        self.B_proj = nn.Linear(d_model, d_state)
        self.delta_proj = nn.Linear(d_model, d_state)

        # SSM output: project hidden state back to d_model
        self.C_out = nn.Linear(d_state, d_model)

        # Output gate
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Convolution
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq)
        x_conv = self.conv1d(x_conv)[..., : x.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.silu(x_conv)

        # SSM
        delta = self.delta_proj(x_conv)  # (batch, seq, d_state)
        B = self.B_proj(x_conv)  # (batch, seq, d_state)
        u = self.u_proj(x_conv)  # (batch, seq, d_state)

        hs = selective_scan(u, delta, self.A, B)  # (batch, seq, d_state)
        y_ssm = self.C_out(hs)  # (batch, seq, d_model)

        # Gate
        gate = torch.sigmoid(self.gate_proj(x))
        output = self.out_proj(y_ssm * gate)

        return self.dropout(output) + residual


class MambaTimeSeriesModel(nn.Module):
    """Mamba-based time series model for FX prediction.

    Processes long sequences (up to 1024 bars) to produce
    next-bar price predictions.

    Architecture:
      Input (batch, seq, features) -> Linear -> MambaBlock x N -> Head -> prediction
    """

    def __init__(
        self,
        n_features: int = 49,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.01)

        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, 1) next-bar prediction
        """
        batch, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Use last timestep for prediction
        x = x[:, -1, :]
        return self.head(x)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """NumPy-compatible prediction interface.

        Args:
            X: (batch, seq_len, n_features)

        Returns:
            (batch,) predictions
        """
        self.eval()
        x = torch.from_numpy(X).float()
        out = self.forward(x)
        return out.squeeze(-1).cpu().numpy()
