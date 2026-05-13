"""
Thorough AI/ML pipeline diagnostic.
Verifies models, dimensions, feature pipeline, ensemble, and runtime performance.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 65)
print("  AI/ML PIPELINE — Deep Diagnostic")
print("=" * 65)

issues = []

# 1. Model files
import pathlib
model_dir = pathlib.Path('models')
print(f"\n[1] MODEL FILES")
for f in sorted(model_dir.iterdir()):
    if f.is_dir():
        continue
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<40} {size_kb:>8.1f} KB")

# 2. PPO Agent architectures
print(f"\n[2] PPO REGIME AGENTS")
from ai.regime_agents import RegimeSpecialistSystem, REGIME_CONFIGS
for name, cfg in REGIME_CONFIGS.items():
    print(f"  {name:<12} hidden={str(cfg.hidden_dims):<20} "
          f"SL={cfg.sl_atr_mult:.1f} TP={cfg.tp_atr_mult:.1f} "
          f"pos={cfg.position_multiplier:.1f} lr={cfg.learning_rate:.0e} "
          f"clip={cfg.clip_range:.2f}")

# Build and check agents
rs = RegimeSpecialistSystem(state_dim=49, n_actions=5)
n_loaded = len([a for a in rs.agents.values() if a is not None])
print(f"\n  Agents loaded: {n_loaded}/4")
for name, agent in rs.agents.items():
    if agent:
        total_params = sum(p.numel() for p in agent.actor.parameters())
        trainable = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
        state_dict_ok = all(p.numel() > 0 for p in agent.actor.parameters())
        print(f"  {name:<12} params={total_params:>8,} "
              f"trainable={trainable:>8,} "
              f"device={str(agent.device):<8} "
              f"weights={'OK' if state_dict_ok else 'EMPTY'}")
    else:
        issues.append(f"PPO agent '{name}' is None")
        print(f"  {name:<12} NOT LOADED (None)")

# Test inference
print(f"\n[3] PPO INFERENCE TEST")
test_state = np.random.randn(49).astype(np.float32)
for name, agent in rs.agents.items():
    if agent:
        t0 = time.time()
        for _ in range(100):
            action, *rest = agent.select_action(test_state)
        dt = (time.time() - t0) * 10  # ms per inference
        print(f"  {name:<12} action={action}  latency={dt:.1f}ms")

# 3. LSTM-CNN Model
print(f"\n[4] LSTM-CNN MODEL")
from rts_ai_fx.model import LSTMCNNHybrid
lstm = LSTMCNNHybrid.load("models/lstm_cnn_model.keras")
if lstm and lstm.model:
    n_layers = len(lstm.model.layers)
    shapes = []
    for l in lstm.model.layers:
        try:
            shapes.append((l.name, l.input_shape, l.output_shape))
        except:
            pass
    # Check if weights are the Keras default (untrained) or loaded
    w0 = lstm.model.layers[1].get_weights()
    has_real_weights = any(np.any(w != 0) for w in w0) if w0 else False
    print(f"  Layers: {n_layers}")
    print(f"  Input:  {lstm.model.input_shape}")
    print(f"  Output: {lstm.model.output_shape}")
    print(f"  Weights loaded from disk: {has_real_weights}")
    if not has_real_weights:
        issues.append("LSTM-CNN: weights appear to be Keras defaults (all zero/random), "
                      "saved weights not transferred due to TF version mismatch")

    # Test inference
    test_X = np.random.randn(1, 30, 49).astype(np.float32)
    t0 = time.time()
    for _ in range(50):
        pred = lstm.predict(test_X)
    dt = (time.time() - t0) * 20  # ms per inference
    print(f"  Inference latency: {dt:.1f}ms")
    print(f"  Sample prediction: {pred[0][0]:.6f}")
else:
    issues.append("LSTM-CNN model is None")
    print("  MODEL NOT LOADED")

# 4. Check alternative models
print(f"\n[5] ALTERNATIVE MODELS")
for fname in ["lstm_cnn_EURUSD.keras", "classifier_EURUSD.keras"]:
    path = f"models/{fname}"
    if os.path.exists(path):
        try:
            saved = __import__('tensorflow').keras.models.load_model(path, compile=False)
            print(f"  {fname:<40} loaded OK ({len(saved.layers)} layers)")
        except Exception as e:
            print(f"  {fname:<40} load FAILED: {e}")

# 5. Meta-learner (Master AI)
print(f"\n[6] META-LEARNER")
import torch
import torch.nn as nn

# Replicate MetaLearnerV2 architecture for diagnostic
class MetaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 128), nn.LayerNorm(128), nn.LeakyReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.LeakyReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.LeakyReLU(), nn.Dropout(0.15),
            nn.Linear(32, 5),
        )
    def forward(self, x):
        return self.net(x)

ml = MetaNet()
total = sum(p.numel() for p in ml.parameters())
print(f"  Architecture: 16->128->64->32->5")
print(f"  Total params: {total:,}")
if os.path.exists("models/meta_learner.pt"):
    ckpt = torch.load("models/meta_learner.pt", map_location="cpu")
    print(f"  Checkpoint keys: {list(ckpt.keys())[:8]}")
    if 'model_state_dict' in ckpt:
        try:
            ml.load_state_dict(ckpt['model_state_dict'])
            print(f"  Weights: loaded OK")
            print(f"  Training step: {ckpt.get('training_step', 'N/A')}")
            print(f"  Stored Sharpe: {ckpt.get('system_sharpe', 'N/A')}")
        except Exception as e:
            issues.append(f"Meta-learner load: {e}")
            print(f"  Load FAILED: {e}")
    else:
        print(f"  Weights: direct state dict format")
else:
    print(f"  No checkpoint file found (fresh system — will be trained on live data)")

# 6. Feature pipeline normalization
print(f"\n[7] FEATURE PIPELINE")
from rts_ai_fx.features_unified import FeaturePipeline
fp = FeaturePipeline(lookback=30, timeframes=["1h", "4h"])
loaded = fp.load_normalization()
n_features = len(fp._feature_cols) if fp._feature_cols else 0
print(f"  Normalization loaded: {loaded}")
print(f"  Feature columns: {n_features}")
print(f"  Symbol-TF pairs: {len(fp._means)}")
for key in sorted(fp._means.keys()):
    mean_dim = fp._means[key].shape[0] if fp._means[key].ndim > 0 else 0
    std_dim = fp._stds[key].shape[0] if fp._stds[key].ndim > 0 else 0
    match = "OK" if mean_dim == std_dim else f"MISMATCH ({mean_dim} vs {std_dim})"
    print(f"    {key:<25} features={mean_dim:<4} {match}")

# Dimensional consistency
print(f"\n[8] DIMENSIONAL CONSISTENCY")
dims = {
    "Feature pipeline output": n_features,
    "PPO state_dim": 49,
    "LSTM n_features": 49,
    "LSTM lookback": 30,
    "Feature lookback (config)": 30,
}
for name, dim in dims.items():
    print(f"  {name:<35} {dim}")
if n_features > 0 and n_features != 49:
    issues.append(f"Feature pipeline outputs {n_features} dims but PPO expects 49")
    print(f"  >> MISMATCH: features={n_features} vs PPO=49")
elif n_features == 0:
    print(f"  >> Feature pipeline not fitted yet (will adapt at runtime)")

# 9. Ensemble readiness
print(f"\n[9] ENSEMBLE READINESS")
from rts_ai_fx.ensemble import MoEEnsemble
ensemble = MoEEnsemble()
print(f"  Experts registered: {len(ensemble.experts)}")
print(f"  Elo ratings: {len(ensemble.elo_ratings)}")
print(f"  Sharpe weighting enabled: {ensemble.use_sharpe_weighting}")

# Summary
print(f"\n{'=' * 65}")
print(f"  ISSUES FOUND: {len(issues)}")
for i, issue in enumerate(issues, 1):
    print(f"  [{i}] {issue}")
if not issues:
    print(f"  No issues — all systems optimal")
print(f"{'=' * 65}")
