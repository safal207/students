#!/usr/bin/env python3
"""Verify all 66 illustrated notebooks without running legacy code.

    python tools/verify_notebook_design.py --workspace /path/notebook-refresh
    python tools/verify_notebook_design.py --repo-root .

Repository CI requires tools/notebook-design-audit.json, copied from the reviewed
build-results.json. It fixes the approved code and illustration hashes outside
the notebooks. Local workspace mode additionally compares every original cell
against source/{id}.ipynb after the explicit credential sanitizer.
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import re
from pathlib import Path

import nbformat
from PIL import Image

from notebook_sanitization import sanitize_notebook
from style_notebooks import (
    ALLOWED_SOURCE_CHANGES, SCHEMA, TEXT_LIMIT, bound_outputs, canonical, digest,
    load_designs, load_json, media_hashes, read_baseline, source_url, text,
    validate_design, write_json,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_one(path: Path, entry: dict, design: dict, manifest: dict,
               audit: dict, baseline: nbformat.NotebookNode | None) -> dict:
    identity = entry["id"]
    # The JSON writer stores multiline text as lists; nbformat.read rejoins them
    # before source and rich-output hashes are compared with the in-memory build.
    notebook = nbformat.read(path, as_version=4)
    try:
        nbformat.validate(notebook)
    except Exception:
        # Validation errors can embed complete cells; never emit their payload.
        raise ValueError("notebook schema validation failed") from None
    metadata = notebook.metadata.get("ai_path_design", {})
    require(metadata.get("schema") == SCHEMA, "presentation schema missing")
    require(metadata.get("id") == identity, "ID mismatch")
    require(metadata.get("course") == entry["course"], "course mismatch")
    require(metadata.get("status") == entry["status"], "archive/starter status mismatch")
    require(metadata.get("execution") == "not-executed-by-presentation-builder", "execution provenance missing")
    source = metadata.get("source", {})
    for key, expected in (("repository", manifest["repository"]), ("ref", manifest["ref"]),
                          ("path", entry["path"]), ("blob", entry.get("blob"))):
        require(source.get(key) == expected, f"source {key} mismatch")
    require(source.get("url") == source_url(manifest, entry), "source URL mismatch")
    require(metadata.get("designSha256") == digest(canonical(design)), "design hash mismatch")
    count = entry["cells"]
    validate_design(design, count)
    require(metadata.get("reviewedCells") == list(range(count)), "source review coverage incomplete")
    require(metadata.get("originalCellCount") == count, "original cell count mismatch")
    originals, generated = [], []
    identifiers = []
    for cell in notebook.cells:
        identifiers.append(cell.get("id"))
        original = cell.metadata.get("ai_path_original")
        new = cell.metadata.get("ai_path_generated")
        require(bool(original) != bool(new), "cell must be original or generated")
        (originals if original else generated).append(cell)
    require(len(identifiers) == len(set(identifiers)), "cell IDs are not unique")
    require([c.metadata.ai_path_original.index for c in originals] == list(range(count)),
            "original cells reordered, removed or duplicated")
    records = metadata.get("sourceCells", [])
    require(len(records) == count, "source audit incomplete")
    code_records = []
    changed = []
    for index, (cell, record) in enumerate(zip(originals, records)):
        require(cell.id == f"src-{identity}-{index:04d}", "original cell ID not deterministic")
        require(record.get("index") == index and record.get("type") == cell.cell_type, "source audit order mismatch")
        actual_hash = digest(text(cell.source))
        require(record.get("sha256") == actual_hash, f"source hash changed at cell {index}")
        require(cell.metadata.ai_path_original.get("sourceSha256") == actual_hash, "cell source audit mismatch")
        if actual_hash != record.get("originalSha256"):
            changed.append(index)
            require(index in ALLOWED_SOURCE_CHANGES.get(identity, set()), "unapproved source exception")
        if cell.cell_type == "code":
            code_records.append(record)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                require(len(text(output.get("text", ""))) <= TEXT_LIMIT, "oversized stream output")
            if "text/plain" in output.get("data", {}):
                require(len(text(output.data["text/plain"])) <= TEXT_LIMIT, "oversized plain-text output")
    require(changed == metadata.get("sourceChangeExceptions"), "source exception audit mismatch")
    require(len(code_records) == entry["codeCells"] == metadata.get("originalCodeCellCount"), "code cell count changed")
    code_hash = digest(canonical([r["sha256"] for r in code_records]))
    original_hash = digest(canonical([r["originalSha256"] for r in code_records]))
    require(code_hash == metadata.get("codeSha256") == audit.get("codeSha256"), "approved code hash mismatch")
    require(original_hash == metadata.get("originalCodeSha256") == audit.get("originalCodeSha256"),
            "original code provenance mismatch")
    require(changed == audit.get("sourceChangeExceptions"), "approved exception mismatch")
    require(metadata.get("sanitization") == audit.get("sanitization"), "approved sanitization notes mismatch")
    require(metadata.get("attachments") == audit.get("attachments"), "approved illustration audit mismatch")
    require(digest(canonical(originals)) == audit.get("preservedCellsSha256"),
            "approved original cell/output/metadata content changed")
    require(digest(canonical({key: value for key, value in notebook.metadata.items() if key != "ai_path_design"})) ==
            audit.get("originalMetadataSha256"), "approved original notebook metadata changed")
    require(digest(canonical([{ "id": cell.id, "source": text(cell.source)} for cell in generated])) ==
            audit.get("generatedMarkdownSha256"), "approved teaching content changed")
    original_only = nbformat.v4.new_notebook(cells=originals)
    require(media_hashes(original_only) == metadata.get("originalMedia"), "original image/audio/HTML changed")
    if baseline is not None:
        expected, _ = sanitize_notebook(copy.deepcopy(baseline), identity)
        expected = nbformat.from_dict(expected)
        require(len(expected.cells) == len(originals), "baseline cell count changed")
        for index, (before, after, saved) in enumerate(zip(baseline.cells, expected.cells, originals)):
            require(text(after.source) == text(saved.source), f"baseline source differs at cell {index}")
            require(digest(text(before.source)) == records[index]["originalSha256"], "baseline source hash differs")
        expected_compaction = bound_outputs(expected, source_url(manifest, entry))
        require(expected_compaction == metadata.get("textOutputCompaction"), "output compaction audit differs")
        for expected_cell, actual_cell in zip(expected.cells, originals):
            require(canonical(expected_cell.get("outputs", [])) == canonical(actual_cell.get("outputs", [])),
                    "original outputs differ beyond documented text compaction")
            require(canonical(expected_cell.get("attachments", {})) == canonical(actual_cell.get("attachments", {})),
                    "original attachments differ")
    expected_roles = ["cover", "preparation", "contents", "concept"] + [
        f"section-{i:02d}" for i in range(1, len(design["steps"]) + 1)
    ] + ["practice", "provenance"]
    roles = [cell.metadata.ai_path_generated.get("role") for cell in generated]
    require(roles == expected_roles, "teaching layer incomplete or reordered")
    for cell, role in zip(generated, expected_roles):
        require(cell.cell_type == "markdown", "generated layer must be Markdown")
        require(cell.id == f"ai-{identity}-{role}", "generated cell ID not deterministic")
        require(not re.search(r"!\[[^\]]*\]\(https?://|<img\b[^>]*src=[\"']https?://", text(cell.source), re.I),
                "new artwork uses external image URL")
    joined = "\n".join(text(cell.source) for cell in generated)
    anchors = re.findall(r'<a\s+id="([^"]+)"\s*>', joined)
    links = re.findall(r'\]\(#([^\s)]+)\)', joined)
    require(len(anchors) == len(set(anchors)), "duplicate internal anchors")
    require(set(links) == set(anchors) and len(links) == len(anchors), "table of contents has broken or missing targets")
    require(f"# {design['title']}" in text(generated[0].source), "lesson title missing")
    require(design["goal"] in text(generated[1].source), "lesson goal missing")
    concept_text = text(next(c.source for c in generated if c.metadata.ai_path_generated.role == "concept"))
    for i, node in enumerate(design["diagram"]["nodes"], 1):
        require(f"{i}. **{node['label']}** — {node['detail']}" in concept_text,
                "readable text equivalent of diagram step missing")
    require("Архивный" in text(generated[0].source) if entry["status"] == "archive" else
            "CPU" in text(generated[0].source), "archive/starter notice missing")
    for i, section in enumerate(design["steps"], 1):
        section_cell = next(c for c in generated if c.id == f"ai-{identity}-section-{i:02d}")
        position = notebook.cells.index(section_cell)
        require(notebook.cells[position + 1].metadata.get("ai_path_original", {}).get("index") == section["beforeCell"],
                "teaching section inserted at wrong original cell")
        require(section["title"] in section_cell.source and section["explanation"] in section_cell.source,
                "content-specific explanation missing")
    attachment_records = {item["name"]: item for item in metadata["attachments"]}
    require(len(attachment_records) == 2, "expected cover and concept diagram")
    found = {}
    for cell in generated:
        references = re.findall(r'\]\(attachment:([^\s)]+)\)', text(cell.source))
        require(set(references) == set(cell.get("attachments", {})), "attachment reference mismatch")
        for name, bundle in cell.get("attachments", {}).items():
            require(name in attachment_records, "unexpected new attachment")
            record = attachment_records[name]
            require(list(bundle) == [record["mime"]], "attachment MIME mismatch")
            try:
                payload = base64.b64decode(text(bundle[record["mime"]]), validate=True)
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                with Image.open(io.BytesIO(payload)) as image:
                    width, height = image.size
                    format_name = image.format
            except Exception:
                raise ValueError("invalid embedded image") from None
            require(1000 <= len(payload) <= 2_000_000, "new illustration byte budget exceeded")
            require(len(payload) == record["bytes"] and digest(payload) == record["sha256"], "attachment hash/size mismatch")
            require([width, height] == [record["width"], record["height"]], "attachment dimensions mismatch")
            require(format_name == {"image/png": "PNG", "image/jpeg": "JPEG"}.get(record["mime"]), "image format mismatch")
            require(width >= 1000 and height >= 500, "illustration resolution too small")
            found[name] = record
    require(set(found) == set(attachment_records), "new illustration missing")
    concept = found.get(f"concept-{identity}.png")
    require(bool(concept), "notebook-specific concept diagram missing")
    return {"id": identity, "status": "PASS", "sourceCells": count, "codeCells": len(code_records),
            "attachments": len(found), "conceptSha256": concept["sha256"],
            "baselineCompared": baseline is not None, "execution": "NOT_RUN"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve() if args.workspace else None
    repo = args.repo_root.resolve()
    manifest_path = args.manifest or (workspace / "source" / "manifest.json" if workspace else
                                     repo / "tools" / "notebook-manifest.json")
    manifest = load_json(manifest_path)
    designs = load_designs(workspace, repo, args.design)
    audit_path = args.audit or (workspace / "review" / "build-results.json" if workspace else
                               repo / "tools" / "notebook-design-audit.json")
    audits = {entry["id"]: entry for entry in load_json(audit_path)["notebooks"]}
    entries = manifest["notebooks"]
    require(len(entries) == len(designs) == len(audits) == 66, "exactly 66 designs and audit records required")
    require({item["id"] for item in entries} == set(designs) == set(audits), "notebook ID sets differ")
    require(sum(item["status"] == "archive" for item in entries) == 65, "archive count changed")
    require(sum(item["status"] == "starter" for item in entries) == 1, "starter count changed")
    results, failures = [], []
    for entry in entries:
        identity = entry["id"]
        path = workspace / "styled" / f"{identity}.ipynb" if workspace else repo / entry["path"]
        try:
            baseline = read_baseline(entry, manifest, workspace, repo, None) if workspace else None
            result = verify_one(path, entry, designs[identity], manifest, audits[identity], baseline)
            results.append(result)
        except ValueError as exc:
            # Only intentional validation messages: they never contain cell content.
            failures.append({"id": identity, "reason": str(exc)[:200]})
        except Exception as exc:
            failures.append({"id": identity, "reason": f"verification error: {type(exc).__name__}"})
    unique = {item["conceptSha256"] for item in results}
    if len(unique) != len(results):
        failures.append({"id": "all", "reason": "concept diagrams are not unique per notebook"})
    report = {"schema": SCHEMA, "status": "PASS" if not failures else "FAIL", "checked": len(results),
              "expected": 66, "uniqueConceptDiagrams": len(unique), "legacyExecution": "NOT_RUN",
              "notebooks": results, "failures": failures}
    report_path = args.report or (workspace / "review" / "design-verification.json" if workspace else
                                 repo / "reports" / "notebook-design" / "verification.json")
    write_json(report_path, report)
    print(f"DESIGN {report['status']}: {len(results)}/66; unique diagrams {len(unique)}; legacy execution NOT_RUN")
    for failure in failures:
        print(f"{failure['id']}: {failure['reason']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
