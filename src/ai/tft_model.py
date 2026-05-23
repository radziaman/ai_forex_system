"""
Temporal Fusion Transformer (TFT) — simplified PyTorch implementation.

Reference: Lim et al. 2021, "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting".

Features:
  - Variable selection gating (static + temporal context)
  - Multi-head self-attention over the temporal dimension
  - Position-wise feed-forward with residual connections & dropout
  - Gated residual network (GRN) blocks
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """GRN: Dense → ELU → Dense + GLU gating + skip connection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.output_size = output_size or input_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_dense = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.hidden_dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_dense = nn.Linear(hidden_size, self.output_size * 2)

        # Context enrichment (static context added to hidden layer)
        self.context_dense: Optional[nn.Linear] = None
        if context_size is not None:
            self.context_dense = nn.Linear(context_size, hidden_size, bias=False)

        # Skip connection projection if dimensions differ
        self.skip: Optional[nn.Linear] = None
        if input_size != self.output_size:
            self.skip = nn.Linear(input_size, self.output_size)

        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        hidden = self.input_dense(x)
        if self.context_dense is not None and context is not None:
            # context may be (B, C) — expand to (B, 1, C) if x is (B, T, D)
            if x.dim() == 3 and context.dim() == 2:
                context = context.unsqueeze(1)
            hidden = hidden + self.context_dense(context)
        hidden = self.elu(hidden)
        hidden = self.hidden_dense(hidden)
        hidden = self.dropout(hidden)
        output = self.output_dense(hidden)
        # GLU gating: split into two halves, sigmoid one, multiply
        if output.size(-1) % 2 != 0:
            raise ValueError("GRN output must be even for GLU split")
        gate = torch.sigmoid(output[..., : self.output_size])
        activated = output[..., self.output_size :] * gate
        return self.layer_norm(activated + residual)


class VariableSelectionNetwork(nn.Module):
    """Learns a sparse combination of input variables using a static context gate."""

    def __init__(
        self,
        num_inputs: int,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_size = input_size
        self.hidden_size = hidden_size

        # One GRN per variable (projects raw variable to hidden)
        self.single_variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
                for _ in range(num_inputs)
            ]
        )

        # Weights generation (softmax over variables)
        self.flattened_grn = GatedResidualNetwork(
            num_inputs * input_size,
            hidden_size,
            num_inputs,
            dropout,
            context_size=context_size,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, num_inputs, input_size) or (B, num_inputs, input_size)
            context: (B, context_size)

        Returns:
            selected: (B, T, hidden_size) or (B, hidden_size)
            weights:  (B, T, num_inputs) or (B, num_inputs)
        """
        *batch_dims, num_inputs, input_size = x.shape
        if num_inputs != self.num_inputs or input_size != self.input_size:
            raise ValueError(
                f"Expected ({self.num_inputs}, {self.input_size}), got ({num_inputs}, {input_size})"
            )

        # Flatten variable dimension for weight generation
        flat = x.reshape(*batch_dims, num_inputs * input_size)
        logits = self.flattened_grn(flat, context)
        weights = F.softmax(logits, dim=-1)

        # Transformed per-variable
        transformed = torch.stack(
            [grn(x[..., i, :]) for i, grn in enumerate(self.single_variable_grns)],
            dim=-2,
        )  # (..., num_inputs, hidden_size)

        # Weighted combination
        weighted = transformed * weights.unsqueeze(-1)
        selected = weighted.sum(dim=-2)  # (..., hidden_size)
        return self.layer_norm(selected), weights


class TemporalFusionTransformer(nn.Module):
    """Simplified TFT for directional forex prediction.

    Architecture:
      1. Static context encoder (symbol + session embedding)
      2. Variable selection over temporal inputs
      3. LSTM encoder-decoder (local processing)
      4. Multi-head self-attention (long-range dependencies)
      5. Position-wise FF + output head
    """

    def __init__(
        self,
        input_size: int = 49,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        static_size: int = 8,
        num_variables: int = 7,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.static_size = static_size
        self.num_variables = num_variables

        # Static context encoder
        self.static_encoder = nn.Linear(static_size, hidden_size)

        # Variable selection network (treats each input channel as a variable)
        self.variable_selection = VariableSelectionNetwork(
            num_inputs=num_variables,
            input_size=input_size // num_variables if num_variables > 0 else input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        # LSTM encoder-decoder for local temporal processing
        self.lstm_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.lstm_decoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Multi-head self-attention for long-range temporal dependencies
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_size, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.attention_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        self.attention_ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.attention_ff_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )

        # Output head: projects last timestep to directional probability
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout, context_size=hidden_size
        )
        self.final_dense = nn.Linear(hidden_size, output_size)

        self._device = torch.device("cpu")
        self.to(self._device)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    def set_device(self, device: torch.device):
        self._device = device
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size) temporal features
            static_context: (B, static_size) symbol encoding + session info

        Returns:
            logits: (B, output_size) raw prediction values
        """
        batch_size, seq_len, _ = x.shape
        x = self._to_device(x)

        # Static context
        if static_context is not None:
            static_context = self._to_device(static_context)
            static_vec = self.static_encoder(static_context)
        else:
            static_vec = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Reshape x into variables for variable selection
        # (B, T, num_variables, input_size // num_variables)
        var_input_size = self.input_size // self.num_variables
        x_vars = x.reshape(batch_size, seq_len, self.num_variables, var_input_size)

        # Variable selection
        selected, _ = self.variable_selection(x_vars, context=static_vec)
        # selected: (B, T, hidden_size)

        # LSTM encoder-decoder (local processing)
        enc_out, _ = self.lstm_encoder(selected)
        dec_out, _ = self.lstm_decoder(selected)
        lstm_out = torch.cat([enc_out, dec_out], dim=-1)  # (B, T, hidden_size)

        # Multi-head self-attention layers
        attn_out = lstm_out
        for attn, norm, ff, ff_norm in zip(
            self.attention_layers,
            self.attention_norms,
            self.attention_ff,
            self.attention_ff_norms,
        ):
            attn_res, _ = attn(attn_out, attn_out, attn_out)
            attn_out = norm(attn_out + attn_res)
            ff_out = ff(attn_out)
            attn_out = ff_norm(attn_out + ff_out)

        # Use last timestep for prediction
        last = attn_out[:, -1, :]  # (B, hidden_size)

        # Enrich with static context
        enriched = self.output_grn(last, context=static_vec)

        # Output head (linear, no activation — caller can apply sigmoid/tanh)
        logits = self.final_dense(enriched)
        return logits

    def predict(
        self,
        x: np.ndarray,
        static_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convenience method: numpy in, numpy out.

        Returns probabilities via sigmoid (for binary/directional tasks).
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32)
            if x_t.dim() == 2:
                x_t = x_t.unsqueeze(0)
            static_t: Optional[torch.Tensor] = None
            if static_context is not None:
                static_t = torch.tensor(static_context, dtype=torch.float32)
                if static_t.dim() == 1:
                    static_t = static_t.unsqueeze(0)
            logits = self.forward(x_t, static_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def save(self, path: str):
        """Save model weights and hyper-parameters."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "static_size": self.static_size,
            "num_variables": self.num_variables,
            "state_dict": self.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "TemporalFusionTransformer":
        """Load model from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            input_size=state["input_size"],
            hidden_size=state["hidden_size"],
            num_heads=state["num_heads"],
            num_layers=state["num_layers"],
            output_size=state["output_size"],
            dropout=state["dropout"],
            static_size=state["static_size"],
            num_variables=state["num_variables"],
        )
        instance.load_state_dict(state["state_dict"])
        instance.eval()
        return instance
