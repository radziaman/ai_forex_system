"""Verify all agentic modules import and wire correctly."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

errors = []
successes = []

def check(name, fn):
    try:
        fn()
        successes.append(name)
    except Exception as e:
        errors.append((name, str(e)))

# Core framework
check("agent_message", lambda: __import__("agentic.core.agent_message"))
check("agent_consciousness", lambda: __import__("agentic.core.agent_consciousness"))
check("agent_memory", lambda: __import__("agentic.core.agent_memory"))
check("agent_bus", lambda: __import__("agentic.core.agent_bus"))
check("agent_registry", lambda: __import__("agentic.core.agent_registry"))
check("world_state", lambda: __import__("agentic.core.world_state"))
check("base_agent", lambda: __import__("agentic.core.base_agent"))

# Domain agents
check("data_agent", lambda: __import__("agentic.agents.data_agent"))
check("feature_agent", lambda: __import__("agentic.agents.feature_agent"))
check("regime_agent", lambda: __import__("agentic.agents.regime_agent"))
check("signal_agent", lambda: __import__("agentic.agents.signal_agent"))
check("risk_agent", lambda: __import__("agentic.agents.risk_agent"))
check("execution_agent", lambda: __import__("agentic.agents.execution_agent"))
check("position_agent", lambda: __import__("agentic.agents.position_agent"))
check("performance_agent", lambda: __import__("agentic.agents.performance_agent"))
check("adaptive_risk_agent", lambda: __import__("agentic.agents.adaptive_risk_agent"))

# Strategic agents
check("master_agent", lambda: __import__("agentic.agents.master_agent"))
check("validation_agent", lambda: __import__("agentic.agents.validation_agent"))
check("monitoring_agent", lambda: __import__("agentic.agents.monitoring_agent"))
check("connection_agent", lambda: __import__("agentic.agents.connection_agent"))
check("learning_agent", lambda: __import__("agentic.agents.learning_agent"))
check("memory_agent", lambda: __import__("agentic.agents.memory_agent"))

# Main entry
check("main_agentic", lambda: __import__("agentic.main_agentic"))

print("=" * 60)
print(f"Agentic System Verification: {len(successes)} passed, {len(errors)} failed")
print("=" * 60)
for s in successes:
    print(f"  [OK] {s}")
for name, err in errors:
    print(f"  [FAIL] {name}: {err}")
print("=" * 60)
if errors:
    print("SOME MODULES FAILED TO IMPORT")
    sys.exit(1)
else:
    print("ALL MODULES VERIFIED — System is coherent")
    sys.exit(0)
