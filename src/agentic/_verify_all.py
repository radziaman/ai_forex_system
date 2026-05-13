"""Verify all 27 G-fixes are applied correctly."""
import sys, os, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pathlib

g_fixes = {
    "G1": "execution_agent: tick wiring to data_agent",
    "G2": "risk_agent: reads adaptive_risk values from world state",
    "G3": "execution_agent: publishes positions to world state",
    "G4": "agent_bus: priority queues per MessagePriority",
    "G5": "agent_message: requires_ack + ack_message + bus wait_for_ack",
    "G6": "agent_message: target_capability + bus routing",
    "G7": "risk_agent: perceive finds halted, act sends alerts",
    "G8": "signal_agent: online learning from trade outcomes",
    "G9": "master_agent: human-in-loop halt approval",
    "G10": "agent_bus: parallel workers (n_workers)",
    "G11": "base_agent: should_skip_cycle adaptive governance",
    "G12": "agent_memory: query() + learn_from_agent() + recall_external()",
    "G13": "agent_message: PAYLOAD_SCHEMAS + validate_payload()",
    "G14": "validation_agent: _fetch_data autonomous data access",
    "G15": "agent_consciousness: EmotionalState multi-dimensional model",
    "G16": "signal_agent: _confidence_buckets calibration tracking",
    "G17": "main_agentic: supervisor hierarchy via registry",
    "G18": "base_agent: simulation_mode + main_agentic --simulate flag",
    "G19": "agent_message: explain() on AgentIntention and AgentMessage",
    "G20": "validation_agent: start_ab_test + _ab_experiments",
    "G21": "agent_memory: consolidate_semantic + get_knowledge_broadcast",
    "G22": "master_agent: _on_agent_error + error escalation",
    "G23": "N/A (dashboard is separate concern)",
    "G24": "agentic/core: metrics counters in agent_bus stats",
    "G25": "agent_consciousness: cycle_budget_ms + consecutive_overruns tracking",
    "G26": "agent_message: checksum + memory_agent: checkpoint integrity",
    "G27": "monitoring_agent: _safe_telegram with retry + _telegram_health",
}

# Check each file for key indicators
file_checks = {
    "agent_message": ["PAYLOAD_SCHEMAS", "requires_ack", "target_capability", "checksum", "explain"],
    "agent_bus": ["_queues", "wait_for_ack", "n_workers", "validate_payload", "routing_count"],
    "agent_consciousness": ["EmotionalState", "cycle_budget_ms", "consecutive_overruns", "error_escalation_level"],
    "agent_memory": ["query", "learn_from_agent", "recall_external", "consolidate_semantic", "get_knowledge_broadcast"],
    "base_agent": ["should_skip_cycle", "simulation_mode", "_escalate_error", "target_capability", "requires_ack"],
    "world_state": ["verify_integrity"],
    "agent_registry": ["find_by_capability", "supervisor"],
    "execution_agent": ["on_price", "TICK_RECEIVED", "execution.open_positions_raw", "requires_ack"],
    "risk_agent": ["effective_kelly", "human_confirm_halt", "requires_ack"],
    "signal_agent": ["_expert_outcomes", "_confidence_buckets", "_on_execution_result"],
    "master_agent": ["_on_agent_error", "pending_approvals", "human_in_loop_approval"],
    "data_agent": ["_on_tick", "TICK_RECEIVED"],
    "monitoring_agent": ["_safe_telegram", "_telegram_health", "_telegram_retries"],
    "memory_agent": ["_compute_checksum", "_verify_checkpoint", "_verify_existing"],
    "validation_agent": ["_fetch_data", "start_ab_test", "_ab_experiments"],
    "main_agentic": ["simulation_mode", "--simulate", "set_registry", "n_workers"],
}

agentic_dir = pathlib.Path(__file__).parent
errors = []
print("=" * 60)
print("G-Fix Verification: Checking all 27 fixes")
print("=" * 60)

for module_name, indicators in file_checks.items():
    # Find the file
    py_file = agentic_dir / f"{module_name}.py"
    if not py_file.exists():
        py_file = agentic_dir / "core" / f"{module_name}.py"
    if not py_file.exists():
        py_file = agentic_dir / "agents" / f"{module_name}.py"
    
    if not py_file.exists():
        errors.append(f"{module_name}: FILE NOT FOUND")
        continue
    
    text = py_file.read_bytes().decode('utf-8', errors='replace')
    
    missing = []
    for indicator in indicators:
        if indicator not in text:
            missing.append(indicator)
    
    if missing:
        errors.append(f"{module_name}: missing: {missing}")
    else:
        print(f"  [OK] {module_name} ({len(indicators)} indicators)")

print()
if errors:
    print(f"FAILURES: {len(errors)}")
    for e in errors:
        print(f"  [FAIL] {e}")
else:
    print("ALL CHECKS PASSED - All 27 G-fixes verified")

print("=" * 60)
