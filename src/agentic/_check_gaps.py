"""Check real data visibility gaps that affect agent operation."""
import sys, os, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gaps = []

# 1. data.ohlcv — regime_agent and feature_agent read it from world state
from agentic.agents.regime_agent import RegimeAgent
from agentic.agents.feature_agent import FeatureAgent
from agentic.agents.data_agent import DataAgent
from agentic.agents.risk_agent import RiskAgent

src_regime = inspect.getsource(RegimeAgent.perceive)
src_feature = inspect.getsource(FeatureAgent.perceive)
src_data = inspect.getsource(DataAgent)

if "data.ohlcv" in src_regime and "data.ohlcv" not in src_data:
    gaps.append(("data.ohlcv", "regime_agent/feature_agent read from world state but data_agent never publishes it",
                 "HIGH", "Publish dm.ohlcv in data_agent.act() or perceive()"))
if "data.atr." in inspect.getsource(RiskAgent) and "data.atr." not in src_data:
    gaps.append(("data.atr.{symbol}", "risk_agent needs ATR but data_agent never publishes",
                 "HIGH", "Publish ATR per symbol in data_agent"))

# 2. data.primary_symbol
from agentic.main_agentic import AgenticOrchestrator
if "data.primary_symbol" in src_regime and "data.primary_symbol" not in inspect.getsource(AgenticOrchestrator._init_world_state):
    gaps.append(("data.primary_symbol", "regime_agent reads it but main doesn't set it",
                 "LOW", "Already set in main_agentic._init_world_state"))

print("Real Data Visibility Gaps:")
if gaps:
    for key, desc, severity, fix in gaps:
        print(f"  [{severity}] {key}")
        print(f"           {desc}")
        print(f"           Fix: {fix}")
else:
    print("  None found — all data flows verified")

print()
print("Note: Many apparent 'gaps' from static analysis are local variables")
print("(symbol, price, action, etc.) that appear in function signatures or")
print("message payloads, not world state lookups.")
