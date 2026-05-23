"""Trace all external (non-agentic) local imports from agentic modules."""

import sys
import os
import ast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
import pathlib  # noqa: E402

agentic_dir = pathlib.Path(__file__).parent

# All local src/ packages
src_packages = {
    d.name
    for d in pathlib.Path("src").iterdir()
    if d.is_dir() and d.name != "agentic" and not d.name.startswith("_")
}

all_imports = set()

for py_file in sorted(agentic_dir.rglob("*.py")):  # noqa: C901
    if py_file.name.startswith("_"):
        continue
    text = py_file.read_bytes().decode("utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in src_packages or root == "services":
                    all_imports.add((root, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in src_packages or root == "services":
                    all_imports.add((root, node.module))

print("=== Local source packages imported by agentic system ===")
by_pkg = {}
for root, full in sorted(all_imports):
    by_pkg.setdefault(root, set()).add(full)

for pkg, imports in sorted(by_pkg.items()):
    print(f"\n  {pkg}/")
    for imp in sorted(imports):
        print(f"    ├── {imp}")

total_files = sum(len(v) for v in by_pkg.values())
print(f"\n  Total: {len(by_pkg)} packages, {total_files} import paths")
print(
    f"  Agentic files: {sum(1 for f in agentic_dir.rglob('*.py') if not f.name.startswith('_'))}"  # noqa: E501
)
