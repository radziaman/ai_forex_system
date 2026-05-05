"""
Multi-Timeframe Attention Fusion.
Learn which timeframes to weight for each prediction.
Replaces simple concatenation with attention-based fusion.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from loguru import logger


class TimeframeAttention(nn.Module):
    """
    Attention mechanism to weight different timeframes.
    Learns which timeframes are most relevant for current market conditions.
    """
    
    def __init__(self, feature_dim: int, n_timeframes: int = 4, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_timeframes = n_timeframes
        self.num_heads = num_heads
        
        # Projection layers
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Context vector for global market state
        self.context_proj = nn.Linear(feature_dim * n_timeframes, feature_dim)
        
    def forward(
        self, 
        tf_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            tf_features: List of tensors, each [batch, seq_len, feature_dim]
                        for each timeframe.
        
        Returns:
            fused: [batch, feature_dim] - attention-fused features
            attention_weights: Dict with attention per timeframe
        """
        if not tf_features or len(tf_features) == 0:
            return None, {}
        
        batch_size = tf_features[0].shape[0]
        
        # Stack timeframes: [batch, n_timeframes, seq_len, feature_dim]
        # First, average over seq_len to get timeframe embeddings
        tf_embeddings = []
        for tf in tf_features:
            # Global average pooling over sequence
            tf_emb = tf.mean(dim=1)  # [batch, feature_dim]
            tf_embeddings.append(tf_emb)
        
        # Stack: [batch, n_timeframes, feature_dim]
        stacked = torch.stack(tf_embeddings, dim=1)
        
        # Self-attention across timeframes
        attended, attn_weights = self.mha(
            stacked, stacked, stacked  # Q, K, V are all the timeframes
        )
        
        # Average attention weights for interpretation
        attn_weights = attn_weights.mean(dim=1)  # [batch, n_timeframes]
        attn_dict = {
            f"tf_{i}": attn_weights[0, i].item()
            for i in range(min(len(tf_features), self.n_timeframes))
        }
        
        # Weighted sum of timeframes
        attn_weights = attn_weights.unsqueeze(-1)  # [batch, n_timeframes, 1]
        fused = (attended * attn_weights).sum(dim=1)  # [batch, feature_dim]
        
        # Residual connection
        context = torch.cat(tf_embeddings, dim=-1)  # [batch, n_timeframes * feature_dim]
        context = self.context_proj(context)
        fused = self.output_proj(fused + context)
        
        return fused, attn_dict


class TemporalAttentionFusion(nn.Module):
    """
    Advanced fusion: Combines timeframe attention with temporal attention.
    Also learns to weight features within each timeframe.
    """
    
    def __init__(
        self, 
        timeframes: List[str],
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.timeframes = timeframes
        self.feature_dims = feature_dims
        
        # Per-timeframe feature attention
        self.feature_attention = nn.ModuleDict({
            tf: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for tf, dim in feature_dims.items()
        })
        
        # Timeframe attention
        total_dim = sum(feature_dims.values())
        self.timeframe_attention = TimeframeAttention(
            feature_dim=total_dim,
            n_timeframes=len(timeframes),
        )
        
        # Output projection
        self.output = nn.Linear(total_dim, hidden_dim)
        
    def forward(
        self,
        tf_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            tf_data: Dict mapping timeframe name to tensor [batch, seq_len, features]
        
        Returns:
            fused: [batch, hidden_dim]
            attention_info: Dict with attention weights
        """
        if not tf_data:
            return None, {}
        
        # Apply feature attention per timeframe
        attended_features = []
        feature_attn = {}
        
        for tf in self.timeframes:
            if tf not in tf_data:
                continue
            x = tf_data[tf]  # [batch, seq_len, features]
            
            # Feature attention scores
            scores = self.feature_attention[tf](x)  # [batch, seq_len, 1]
            weights = F.softmax(scores, dim=1)
            
            # Weighted sum of features
            attended = (x * weights).sum(dim=1)  # [batch, features]
            attended_features.append(attended)
            feature_attn[tf] = weights.mean(dim=0).mean(dim=0).item()
        
        if not attended_features:
            return None, {}
        
        # Timeframe attention
        # Stack: [batch, n_timeframes, total_features]
        stacked = torch.stack(attended_features, dim=1)
        
        # Self-attention across timeframes
        batch_size = stacked.shape[0]
        query = stacked  # [batch, n_tf, features]
        
        # Simple dot-product attention
        scores = torch.bmm(query, query.transpose(1, 2))  # [batch, n_tf, n_tf]
        attn_weights = F.softmax(scores, dim=-1)
        fused = torch.bmm(attn_weights, query).mean(dim=1)  # [batch, features]
        
        # Output projection
        output = self.output(fused)
        
        # Attention info for interpretation
        tf_attn_dict = {
            f"tf_attn_{self.timeframes[i]}": attn_weights[0, i, :].mean().item()
            for i in range(len(self.timeframes))
        }
        attention_info = {
            "feature_attention": feature_attn,
            "timeframe_attention": tf_attn_dict,
        }
        
        return output, attention_info


class AttentionFusionPipeline:
    """
    Complete pipeline for multi-timeframe attention fusion.
    Integrates with existing FeaturePipeline.
    """
    
    def __init__(
        self,
        timeframes: List[str] = None,
        lookback: int = 30,
    ):
        self.timeframes = timeframes or ["15m", "1h", "4h"]
        self.lookback = lookback
        self.attention_model: Optional[TemporalAttentionFusion] = None
        self.fusion_weights_history: List[Dict] = []
        
    def init_model(self, feature_dims: Dict[str, int], hidden_dim: int = 256):
        """Initialize the attention fusion model."""
        self.attention_model = TemporalAttentionFusion(
            timeframes=self.timeframes,
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
        )
        logger.info(f"Attention fusion model initialized: {feature_dims}")
        
    def fuse(
        self,
        tf_features: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fuse multi-timeframe features using attention.
        
        Args:
            tf_features: Dict mapping timeframe to numpy array [lookback, n_features]
        
        Returns:
            fused_vector: [hidden_dim] numpy array
            attention_info: Dict with attention weights
        """
        if self.attention_model is None:
            # Fallback to simple concatenation
            vectors = []
            for tf in self.timeframes:
                if tf in tf_features:
                    vectors.append(tf_features[tf].flatten())
            return np.concatenate(vectors) if vectors else np.array([]), {}
        
        # Convert to tensors
        tensors = {}
        for tf in self.timeframes:
            if tf in tf_features and tf in self.attention_model.feature_dims:
                arr = tf_features[tf]
                if len(arr.shape) == 2:  # [lookback, features]
                    tensor = torch.FloatTensor(arr).unsqueeze(0)  # [1, lookback, features]
                else:
                    tensor = torch.FloatTensor(arr).unsqueeze(0)
                tensors[tf] = tensor
        
        # Apply attention fusion
        with torch.no_grad():
            fused_tensor, attn_info = self.attention_model(tensors)
            fused_vector = fused_tensor.squeeze(0).numpy() if fused_tensor is not None else np.array([])
        
        # Store attention history
        self.fusion_weights_history.append(attn_info)
        if len(self.fusion_weights_history) > 100:
            self.fusion_weights_history = self.fusion_weights_history[-100:]
            
        return fused_vector, attn_info
    
    def get_attention_summary(self) -> Dict:
        """Get summary of recent attention weights."""
        if not self.fusion_weights_history:
            return {}
        
        # Average attention over last 10 updates
        recent = self.fusion_weights_history[-10:]
        summary = {
            "recent_timeframe_weights": {},
            "recent_feature_weights": {},
        }
        
        # Average timeframe attention
        tf_weights = {}
        for info in recent:
            tf_attn = info.get("timeframe_attention", {})
            for k, v in tf_attn.items():
                tf_weights[k] = tf_weights.get(k, []) + [v]
        
        summary["recent_timeframe_weights"] = {
            k: float(np.mean(v)) for k, v in tf_weights.items()
        }
        
        return summary
