"""
Verifies every agent's data dependencies are satisfied by the world state / message bus.
Traces reads and writes across all 15 agents.
"""

import sys
import os
import ast
import pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

agentic_dir = pathlib.Path(__file__).parent

# Map: world_state_key -> (writer_agent, reader_agents)
# Built by static analysis of get_world / set_world calls
key_flows = {}  # key -> {"writers": set, "readers": set}

# ──────────────────────────────────────────────────────────────────────
# KNOWN ORPHANS — keys that ARE consumed but the static AST analyzer
# cannot detect because they are consumed via:
#
#   dynamic     – f-string patterns like f"ab_test.{test_id}" that
#                 the static checker cannot resolve to exact keys
#
#   external    – read by non-agent code (ws.get() in _run_live.py,
#                 dashboard bridge, notebooks, etc.)
#
#   monitoring  – status / health keys read by dashboards, health
#                 endpoints, or manual inspection via world.snapshot()
#
# When adding new monitoring keys to agents, add them here too.
# ──────────────────────────────────────────────────────────────────────

_KNOWN_ORPHANS: dict = {
    # Dynamic (f-string variables)
    "ab_test.{sym}": "dynamic",
    "ab_test.{sym}.status": "dynamic",
    "signal.rejected.{sym}": "dynamic",
    # External consumers (ws.get in _run_live.py, dashboard)
    "data.freshness": "external",
    "data.tick_rate": "external",
    "monitoring.telegram_health": "external",
    "strategy_attribution": "external",
    # Monitoring / observability (status, health, diagnostics)
    "adaptive_risk.status": "monitoring",
    "connection.connected": "monitoring",
    "connection.gave_up": "monitoring",
    "connection.resubscribed": "monitoring",
    "connection.retries": "monitoring",
    "connection.status": "monitoring",
    "data.health_pct": "monitoring",
    "data.last_refresh_ts": "monitoring",
    "data.stale_symbols": "monitoring",
    "data.status": "monitoring",
    "data.symbols": "monitoring",
    "execution.mode": "monitoring",
    "execution.status": "monitoring",
    "execution.subscriptions": "monitoring",
    "feature.ready": "monitoring",
    "feature.schema_version": "monitoring",
    "learning.last_training": "monitoring",
    "learning.retraining_count": "monitoring",
    "market.active_sessions": "monitoring",
    "market.is_high_liquidity": "monitoring",
    "market.is_open": "monitoring",
    "market.liquidity_reason": "monitoring",
    "market.sessions": "monitoring",
    "master.directives_issued": "monitoring",
    "master.domain_health": "monitoring",
    "master.health": "monitoring",
    "master.last_diagnostics": "monitoring",
    "master.started_at": "monitoring",
    "master.status": "monitoring",
    "models.ppo_trained": "monitoring",
    "performance.status": "monitoring",
    "persistence.integrity_errors": "monitoring",
    "persistence.path": "monitoring",
    "persistence.status": "monitoring",
    "positions.concentration_warnings": "monitoring",
    "positions.correlation_warnings": "monitoring",
    "positions.status": "monitoring",
    "risk.circuit_breaker_active": "monitoring",
    "risk.halted_symbols": "monitoring",
    "risk.initial_balance": "monitoring",
    "risk.kill_switch_release_requested": "monitoring",
    "risk.stats": "monitoring",
    "risk.status": "monitoring",
    "screening.cross_asset_prices": "monitoring",
    "screening.last_scan": "monitoring",
    "screening.symbols_scanned": "monitoring",
    "signal.active_symbols": "monitoring",
    "signal.drifted_list": "monitoring",
    "signal.experts": "monitoring",
    "signal.models_loaded": "monitoring",
    "validation.last_result": "monitoring",
    "validation.status": "monitoring",
}


def extract_world_keys(filepath: str):  # noqa: C901
    """Extract world state key patterns from agent source."""
    text = pathlib.Path(filepath).read_bytes().decode("utf-8", errors="replace")
    trees = ast.parse(text)

    writers = set()
    readers = set()

    for node in ast.walk(trees):
        # self.set_world("key", value) → writer
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "set_world":
                if node.args and isinstance(node.args[0], ast.Constant):
                    writers.add(node.args[0].value)
            if isinstance(func, ast.Attribute) and func.attr == "update_world":
                if node.args and isinstance(node.args[0], ast.Constant):
                    writers.add(node.args[0].value)
            if isinstance(func, ast.Attribute) and func.attr in ("get_world", "get"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    readers.add(node.args[0].value)
            # f-string patterns: self.get_world(f"data.atr.{symbol}", ...)
            if isinstance(func, ast.Attribute) and func.attr in ("get_world", "get"):
                if node.args and isinstance(node.args[0], ast.JoinedStr):
                    # Extract constant parts of f-strings
                    parts = []
                    for v in node.args[0].values:
                        if isinstance(v, ast.Constant):
                            parts.append(v.value)
                        elif isinstance(v, ast.FormattedValue):
                            parts.append("{sym}")
                    if parts:
                        key_pattern = "".join(parts)
                        if "{sym}" in key_pattern:
                            readers.add(key_pattern)
                            readers.add(key_pattern.replace("{sym}", "*"))

    # f-string patterns with get_world
    for node in ast.walk(trees):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("get_world", "get"):
                if node.args and isinstance(node.args[0], ast.JoinedStr):
                    parts = []
                    for v in node.args[0].values:
                        if isinstance(v, ast.Constant):
                            parts.append(v.value)
                        elif isinstance(v, ast.FormattedValue):
                            parts.append("{sym}")
                    if parts:
                        key_pattern = "".join(parts)
                        readers.add(key_pattern)

    # self.set_world(f"data.spread.{symbol}", ...) patterns
    for node in ast.walk(trees):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "set_world":
                    if node.args and isinstance(node.args[0], ast.JoinedStr):
                        parts = []
                        for v in node.args[0].values:
                            if isinstance(v, ast.Constant):
                                parts.append(v.value)
                            elif isinstance(v, ast.FormattedValue):
                                parts.append("{sym}")
                        if parts:
                            key_pattern = "".join(parts)
                            writers.add(key_pattern)

    return writers, readers


all_writers = {}
all_readers = {}

for py_file in sorted(agentic_dir.rglob("agents/*.py")):
    if py_file.name.startswith("_"):
        continue
    agent_name = py_file.stem
    writers, readers = extract_world_keys(str(py_file))
    all_writers[agent_name] = writers
    all_readers[agent_name] = readers

# Also check main_agentic.py
main_file = agentic_dir / "main_agentic.py"
if main_file.exists():
    writers, readers = extract_world_keys(str(main_file))
    all_writers["main_agentic"] = writers
    all_readers["main_agentic"] = readers

# Build flow map
all_keys = set()
for w in all_writers.values():
    all_keys.update(w)
for r in all_readers.values():
    all_keys.update(r)

print("=" * 80)
print("  DATA FLOW VERIFICATION — Every key written vs read")
print("=" * 80)

gaps = []
orphans = []

for key in sorted(all_keys):  # noqa: C901
    writers = {a for a, ks in all_writers.items() if key in ks}
    readers = {a for a, ks in all_readers.items() if key in ks}

    # Also check pattern matches (e.g., data.spread.{sym} matches data.spread.EURUSD)
    for w, ks in all_writers.items():
        for k in ks:
            if "{sym}" in k:
                pattern = k.replace("{sym}", "*")
                if key.startswith(pattern.rstrip("*")) or key == k.replace(
                    "{sym}", "SYMBOL"
                ):
                    writers.add(w)
    for r, ks in all_readers.items():
        for k in ks:
            if "{sym}" in k:
                if key.startswith(k.replace("{sym}", "").rstrip("*")):
                    readers.add(r)

    if not writers and readers:
        gaps.append((key, readers, "READ BUT NEVER WRITTEN"))
    elif not readers and writers:
        orphans.append(key)
    elif writers and readers:
        writer_list = ", ".join(sorted(writers))
        reader_list = ", ".join(sorted(readers))
        print(f"  [OK] {key:<35} -> {reader_list}")

if gaps:
    print(f"\n  {'='*70}")
    print(f"  GAPS: Data read but never written ({len(gaps)}):")
    print(f"  {'='*70}")
    for key, readers, msg in gaps:
        print(f"  [MISSING] {key:<40} read by {', '.join(sorted(readers))}")
        print(f"             {msg}")

true_orphans = [k for k in orphans if k not in _KNOWN_ORPHANS]
known_orphans = [k for k in orphans if k in _KNOWN_ORPHANS]

if true_orphans:
    print(f"\n  {'='*70}")
    print(f"  ORPHANS: Data written but never read ({len(true_orphans)}):")
    print(f"  {'='*70}")
    for key in sorted(true_orphans):
        writers = {a for a, ks in all_writers.items() if key in ks}
        print(f"  [UNUSED] {key:<40} written by {', '.join(sorted(writers))}")

if known_orphans:
    print(f"\n  {'='*70}")
    print(f"  KNOWN ORPHANS (suppressed — {len(known_orphans)} whitelisted):")
    print(f"  {'='*70}")
    for key in sorted(known_orphans):
        writers = {a for a, ks in all_writers.items() if key in ks}
        cat = _KNOWN_ORPHANS.get(key, "?")
        print(f"  [OK/{cat:<10}] {key:<40} written by {', '.join(sorted(writers))}")

print(
    f"\n  Summary: {len(all_keys)} total keys, "
    f"{len(gaps)} gaps, "
    f"{len(true_orphans)} orphans "
    f"({len(known_orphans)} known false positives suppressed)"
)
print("=" * 80)
