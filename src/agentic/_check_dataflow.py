"""
Verifies every agent's data dependencies are satisfied by the world state / message bus.
Traces reads and writes across all 15 agents.
"""
import sys, os, ast, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

agentic_dir = pathlib.Path(__file__).parent

# Map: world_state_key -> (writer_agent, reader_agents)
# Built by static analysis of get_world / set_world calls
key_flows = {}  # key -> {"writers": set, "readers": set}

def extract_world_keys(filepath: str):
    """Extract world state key patterns from agent source."""
    text = pathlib.Path(filepath).read_bytes().decode('utf-8', errors='replace')
    trees = ast.parse(text)
    
    writers = set()
    readers = set()
    
    for node in ast.walk(trees):
        # self.set_world("key", value) → writer
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == 'set_world':
                if node.args and isinstance(node.args[0], ast.Constant):
                    writers.add(node.args[0].value)
            if isinstance(func, ast.Attribute) and func.attr == 'update_world':
                if node.args and isinstance(node.args[0], ast.Constant):
                    writers.add(node.args[0].value)
            if isinstance(func, ast.Attribute) and func.attr in ('get_world', 'get'):
                if node.args and isinstance(node.args[0], ast.Constant):
                    readers.add(node.args[0].value)
            # f-string patterns: self.get_world(f"data.atr.{symbol}", ...)
            if isinstance(func, ast.Attribute) and func.attr in ('get_world', 'get'):
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
            if isinstance(func, ast.Attribute) and func.attr in ('get_world', 'get'):
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
                if isinstance(func, ast.Attribute) and func.attr == 'set_world':
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
    if py_file.name.startswith('_'):
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

for key in sorted(all_keys):
    writers = {a for a, ks in all_writers.items() if key in ks}
    readers = {a for a, ks in all_readers.items() if key in ks}
    
    # Also check pattern matches (e.g., data.spread.{sym} matches data.spread.EURUSD)
    for w, ks in all_writers.items():
        for k in ks:
            if '{sym}' in k:
                pattern = k.replace('{sym}', '*')
                if key.startswith(pattern.rstrip('*')) or key == k.replace('{sym}', 'SYMBOL'):
                    writers.add(w)
    for r, ks in all_readers.items():
        for k in ks:
            if '{sym}' in k:
                if key.startswith(k.replace('{sym}', '').rstrip('*')):
                    readers.add(r)

    if not writers and readers:
        gaps.append((key, readers, "READ BUT NEVER WRITTEN"))
    elif not readers and writers:
        orphans.append(key)
    elif writers and readers:
        writer_list = ', '.join(sorted(writers))
        reader_list = ', '.join(sorted(readers))
        print(f"  [OK] {key:<35} -> {reader_list}")

if gaps:
    print(f"\n  {'='*70}")
    print(f"  GAPS: Data read but never written ({len(gaps)}):")
    print(f"  {'='*70}")
    for key, readers, msg in gaps:
        print(f"  [MISSING] {key:<40} read by {', '.join(sorted(readers))}")
        print(f"             {msg}")

if orphans:
    print(f"\n  {'='*70}")
    print(f"  ORPHANS: Data written but never read ({len(orphans)}):")
    print(f"  {'='*70}")
    for key in sorted(orphans):
        writers = {a for a, ks in all_writers.items() if key in ks}
        print(f"  [UNUSED] {key:<40} written by {', '.join(sorted(writers))}")

print(f"\n  Summary: {len(all_keys)} total keys, "
      f"{len(gaps)} gaps, {len(orphans)} orphans")
print("=" * 80)
