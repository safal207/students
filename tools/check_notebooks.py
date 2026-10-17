#!/usr/bin/env python3
"""Inventory legacy Jupyter notebooks without executing them.

No cloud access, notebook execution, secret printing, or modification of files.
The new synthetic NumPy exercise can be executed separately with --run-example.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "ML" / "reproducible_mlp_xor_numpy.ipynb"
FILES = sorted(ROOT.rglob("*.ipynb"))
EXPECTED_ARCHIVE = 65  # At the 2026-10-18 baseline, before the new example.
CATEGORY_COUNTS = {
    "CV": 7, "NLP": 10, "AudioNet": 2,
    "Classification": 3, "SeletNetAI": 2,
}


def parse_one(path: Path) -> tuple[bool, int, list[str]]:
    notes: list[str] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, 0, [f"invalid JSON: {exc.__class__.__name__}"]
    cells = raw.get("cells")
    if not isinstance(cells, list):
        return False, 0, ["missing 'cells' list"]
    if not isinstance(raw.get("nbformat"), int) or raw["nbformat"] < 3:
        return False, 0, ["unsupported/missing notebook format"]
    n_code = 0
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") not in ("code", "markdown", "raw"):
            return False, n_code, [f"unsupported cell at index {index}"]
        source = cell.get("source", "")
        if not isinstance(source, (str, list)):
            return False, n_code, [f"invalid cell source at index {index}"]
        if cell["cell_type"] == "code":
            n_code += 1
            if not isinstance(cell.get("outputs", []), list):
                return False, n_code, [f"invalid output list at index {index}"]
            if cell.get("outputs"):
                notes.append("stored outputs (may contain old results or private data)")
    if not raw.get("metadata"):
        notes.append("missing/empty notebook metadata")
    return True, n_code, sorted(set(notes))


def run_example() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install numpy for --run-example: python -m pip install numpy") from exc
    notebook = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            exec(compile(code, f"{EXAMPLE.name}:cell-{index}", "exec"), scope, scope)
    assert scope["test_acc"] > 0.90
    print(f"NEW NOTEBOOK EXECUTION PASS: synthetic test accuracy={scope['test_acc']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-example", action="store_true",
                        help="Execute only the new synthetic MLP example (never legacy notebooks)")
    args = parser.parse_args()
    issues: list[str] = []
    warnings: list[str] = []
    by_dir = Counter()
    cell_count = 0
    for path in FILES:
        rel = path.relative_to(ROOT)
        by_dir[rel.parts[0] if len(rel.parts) > 1 else "(root)"] += 1
        ok, ncode, notes = parse_one(path)
        cell_count += ncode
        if not ok:
            issues.append(f"{rel}: {'; '.join(notes)}")
        elif notes:
            warnings.append(f"{rel}: {'; '.join(notes)}")
    print(f"Found {len(FILES)} notebooks; {cell_count} code cells; {len(issues)} structural errors")
    for folder, count in sorted(by_dir.items()):
        print(f"  {folder}: {count}")
    for note in warnings[:15]:
        print("WARN:", note)
    if len(warnings) > 15:
        print(f"... and {len(warnings) - 15} other notebooks with warnings")
    for issue in issues:
        print("ERROR:", issue)
    if len(FILES) < EXPECTED_ARCHIVE + 1:
        issues.append("Notebook archive unexpectedly smaller than the baseline + example")
    for folder, min_count in CATEGORY_COUNTS.items():
        if by_dir[folder] < min_count:
            issues.append(f"Category {folder} unexpectedly has < {min_count} notebooks")
    if EXAMPLE not in FILES:
        issues.append("New synthetic example notebook missing")
    if issues:
        return 1
    if args.run_example:
        run_example()
    print("STRUCTURE AUDIT PASS. Legacy notebooks were not executed or modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
