#!/usr/bin/env python3
"""Add a portable, illustrated reading layer without executing notebook code.

Local delivery workspace (Windows-safe aliases)::
    python tools/style_notebooks.py --workspace /path/notebook-refresh \
        --art-dir /path/neural-course-cards

Repository checkout (original paths, normally Linux)::
    python tools/style_notebooks.py --repo-root . --art-dir /path/course-covers

Repository mode reads tools/notebook-manifest.json and tools/notebook-design.json.
For a notebook already styled, its original is read from the manifest's pinned
Git commit (local objects only). --source-dir can supply numeric baseline aliases
instead. No network, legacy execution, token printing, or external publishing.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import quote

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ai-path-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import nbformat
from PIL import Image, ImageOps

from notebook_sanitization import sanitize_notebook

SCHEMA = 1
TEXT_LIMIT = 8000
COURSES = {
    "ml": "Нейросети и машинное обучение",
    "ml-python": "Python для машинного обучения",
    "ml-data": "Данные и классификация",
    "ml-vision": "Компьютерное зрение",
    "ml-text": "Нейросети для текста",
    "ml-audio": "Аудио и сигналы",
    "ml-series": "Временные ряды",
    "ml-tuning": "Подбор архитектур и параметров",
}
# Only these source changes are authorised: credentials and inherited Git identity.
ALLOWED_SOURCE_CHANGES = {"064": {18}, "066": {27, 31}}


def text(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_design(design: dict, count: int) -> None:
    identity = design["id"]
    if design.get("reviewedCells") != list(range(count)):
        raise ValueError(f"{identity}: incomplete source-cell review")
    steps = design.get("steps", [])
    # 051 consists of one title and two code cells: two genuine sections suffice.
    minimum_steps = 2 if identity == "051" and count == 3 else 3
    if not minimum_steps <= len(steps) <= 8:
        raise ValueError(f"{identity}: expected {minimum_steps}–8 teaching sections")
    indexes = [item["beforeCell"] for item in steps]
    if indexes != sorted(set(indexes)) or any(not 0 <= i < count for i in indexes):
        raise ValueError(f"{identity}: invalid original-cell section indexes")
    if len(design.get("diagram", {}).get("nodes", [])) != 4:
        raise ValueError(f"{identity}: concept diagram must contain four steps")
    if len(design.get("checks", [])) != 3:
        raise ValueError(f"{identity}: expected three content-specific checks")
    for key, minimum, maximum in (("prerequisites", 2, 4), ("setup", 2, 5), ("evidence", 2, 4)):
        if not minimum <= len(design.get(key, [])) <= maximum:
            raise ValueError(f"{identity}: invalid {key} count")
    if not design.get("limitations") or not design.get("experiment", {}).get("successCriterion"):
        raise ValueError(f"{identity}: missing limitations or experiment criterion")


def load_designs(workspace: Path | None, repo: Path, explicit: Path | None) -> dict[str, dict]:
    path = explicit or repo / "tools" / "notebook-design.json"
    if path.exists():
        raw = load_json(path)
        items = raw["notebooks"] if isinstance(raw, dict) else raw
    elif workspace:
        parts = sorted((workspace / "design").glob("part-*.json"))
        items = [item for part in parts for item in load_json(part)]
    else:
        raise FileNotFoundError("tools/notebook-design.json is required")
    result = {item["id"]: item for item in items}
    if len(result) != len(items):
        raise ValueError("Duplicate notebook design IDs")
    return result


def source_url(manifest: dict, entry: dict) -> str:
    return f"https://github.com/{manifest['repository']}/blob/{manifest['ref']}/{quote(entry['path'], safe='/')}"


def read_baseline(entry: dict, manifest: dict, workspace: Path | None,
                  repo: Path, source_dir: Path | None) -> nbformat.NotebookNode:
    if source_dir or workspace:
        path = (source_dir or workspace / "source") / f"{entry['id']}.ipynb"
        raw = path.read_text(encoding="utf-8-sig")
    else:
        path = repo / entry["path"]
        raw = path.read_text(encoding="utf-8-sig")
        current = json.loads(raw)
        if current.get("metadata", {}).get("ai_path_design"):
            result = subprocess.run(
                ["git", "show", f"{manifest['ref']}:{entry['path']}"], cwd=repo,
                capture_output=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(f"{entry['id']}: pinned baseline unavailable; supply --source-dir")
            raw = result.stdout.decode("utf-8-sig")
    parsed = json.loads(raw)
    if parsed.get("nbformat") != 4:
        raise ValueError(f"{entry['id']}: expected notebook format 4")
    # from_dict avoids incidental conversion of source code; cell IDs are set below.
    notebook = nbformat.from_dict(parsed)
    for cell in notebook.cells:
        cell.source = text(cell.get("source", ""))
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    output.text = text(output.get("text", ""))
                for mime, payload in output.get("data", {}).items():
                    if isinstance(payload, list) and (mime.startswith("text/") or mime.startswith("image/")
                                                      or mime == "application/javascript"):
                        output.data[mime] = "".join(payload)
    return notebook


def packaged_cover(art_dir: Path, course: str) -> bytes:
    candidates = [art_dir / f"{course}{ext}" for ext in (".webp", ".jpg", ".png")]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Missing course artwork: {course}")
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1200, 800), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True, progressive=False)
        return output.getvalue()


def concept_diagram(design: dict) -> bytes:
    """Render a conceptual four-step diagram, never fabricated metric values."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    cream, violet, ink = "#FAF7EE", "#6D4AFF", "#242337"
    palette = [violet, "#087B79", "#D7F33F", "#8050B1"]
    fig = plt.figure(figsize=(14, 7.2), dpi=140, facecolor=cream)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(.65, 6.72, f"AI PATH  /  {design['id']}  /  КАРТА УРОКА", fontsize=11,
            fontweight="bold", color=violet, va="center")
    title = textwrap.fill(design["diagram"]["title"], 64)
    ax.text(.65, 6.23, title, fontsize=22, fontweight="bold", color=ink, va="top", linespacing=1.18)
    positions = [(.65, 3.0), (7.25, 3.0), (.65, .65), (7.25, .65)]
    for number, (node, (x, y), accent) in enumerate(zip(design["diagram"]["nodes"], positions, palette), 1):
        ax.add_patch(FancyBboxPatch((x, y), 6.1, 1.95, boxstyle="round,pad=.02,rounding_size=.16",
                                   linewidth=1.2, edgecolor="#DED8EA", facecolor="white"))
        ax.add_patch(FancyBboxPatch((x + .22, y + 1.31), .48, .4,
                                   boxstyle="round,pad=.03,rounding_size=.09",
                                   linewidth=0, facecolor=accent))
        ax.text(x + .46, y + 1.5, f"{number:02}", ha="center", va="center",
                color=ink if number == 3 else "white", fontweight="bold", fontsize=11)
        ax.text(x + .9, y + 1.64, textwrap.fill(node["label"], 34),
                color=ink, va="top", fontsize=16, fontweight="bold", linespacing=1.12)
        ax.text(x + .3, y + .96, textwrap.fill(node["detail"], 49),
                color="#525167", va="top", fontsize=13, linespacing=1.32)
    for start, finish in (((6.85, 3.95), (7.13, 3.95)), ((6.85, 1.6), (7.13, 1.6))):
        ax.add_patch(FancyArrowPatch(start, finish, arrowstyle="-|>", mutation_scale=17,
                                     color=violet, linewidth=1.7))
    ax.text(.65, .22, "Концептуальная схема • Не результат обучения или вычислений", fontsize=10, color="#625E73")
    output = io.BytesIO()
    fig.savefig(output, format="png", facecolor=cream, dpi=140,
                metadata={"Software": "AI PATH notebook presentation builder"})
    plt.close(fig)
    return output.getvalue()


def attachment_record(name: str, mime: str, payload: bytes) -> dict:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
    return {"name": name, "mime": mime, "bytes": len(payload), "sha256": digest(payload),
            "width": width, "height": height}


def markdown(identity: str, role: str, content: str, attachments: dict | None = None) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(content)
    cell.id = f"ai-{identity}-{role}"
    cell.metadata["ai_path_generated"] = {"schema": SCHEMA, "role": role}
    if attachments:
        cell.attachments = attachments
    return cell


def media_hashes(notebook: nbformat.NotebookNode) -> list[dict]:
    """Track old images/audio/HTML exactly, including alternative MIME renderings."""
    result = []
    for index, cell in enumerate(notebook.cells):
        for output_index, output in enumerate(cell.get("outputs", [])):
            for mime, value in output.get("data", {}).items():
                if mime != "text/plain":
                    result.append({"cell": index, "output": output_index, "mime": mime,
                                   "sha256": digest(canonical(value))})
        for name, bundle in cell.get("attachments", {}).items():
            result.append({"cell": index, "attachment": name, "sha256": digest(canonical(bundle))})
    return result


def compact_text(value: str, url: str) -> str:
    notice = f"\n\n[Архивный вывод сокращён; полный в источнике: {url}]\n\n"
    room = TEXT_LIMIT - len(notice)
    start_limit, end_limit = int(room * .62), room - int(room * .62)
    start, end = value[:start_limit], value[-end_limit:]
    if "\n" in start:
        start = start.rsplit("\n", 1)[0]
    if "\n" in end:
        end = end.split("\n", 1)[-1]
    return start + notice + end


def bound_outputs(notebook: nbformat.NotebookNode, url: str) -> list[dict]:
    changes = []
    for index, cell in enumerate(notebook.cells):
        for output_index, output in enumerate(cell.get("outputs", [])):
            if output.get("output_type") == "stream":
                original = text(output.get("text", ""))
                if len(original) > TEXT_LIMIT:
                    output.text = compact_text(original, url)
                    changes.append({"cell": index, "output": output_index, "mime": "stream",
                                    "originalChars": len(original), "sha256": digest(original)})
            bundle = output.get("data", {})
            if "text/plain" in bundle:
                original = text(bundle["text/plain"])
                if len(original) > TEXT_LIMIT:
                    bundle["text/plain"] = compact_text(original, url)
                    changes.append({"cell": index, "output": output_index, "mime": "text/plain",
                                    "originalChars": len(original), "sha256": digest(original)})
    return changes


def build_one(original: nbformat.NotebookNode, entry: dict, design: dict, manifest: dict,
              cover: bytes) -> tuple[nbformat.NotebookNode, dict]:
    identity, count = entry["id"], len(original.cells)
    validate_design(design, count)
    baseline = copy.deepcopy(original)
    notebook, sanitization = sanitize_notebook(copy.deepcopy(original), identity)
    notebook = nbformat.from_dict(notebook)
    if len(notebook.cells) != count:
        raise ValueError(f"{identity}: sanitizer changed cell count")
    code_audit, source_audit, changed = [], [], []
    for index, (before, after) in enumerate(zip(baseline.cells, notebook.cells)):
        if before.cell_type != after.cell_type:
            raise ValueError(f"{identity}: sanitizer changed cell type {index}")
        after.source = text(after.source)
        before_hash, after_hash = digest(text(before.source)), digest(after.source)
        if before_hash != after_hash:
            if index not in ALLOWED_SOURCE_CHANGES.get(identity, set()):
                raise ValueError(f"{identity}: unapproved source change in cell {index}")
            changed.append(index)
        record = {"index": index, "type": after.cell_type,
                  "originalSha256": before_hash, "sha256": after_hash}
        source_audit.append(record)
        if after.cell_type == "code":
            code_audit.append(record)
        after.id = f"src-{identity}-{index:04d}"
        after.metadata["ai_path_original"] = {"index": index, "sourceSha256": after_hash}
    url = source_url(manifest, entry)
    existing_media = media_hashes(notebook)
    truncated = bound_outputs(notebook, url)
    if media_hashes(notebook) != existing_media:
        raise ValueError(f"{identity}: media altered during text compaction")
    diagram = concept_diagram(design)
    course = COURSES[entry["course"]]
    cover_name, diagram_name = f"ai-path-{entry['course']}.jpg", f"concept-{identity}.png"
    cover_bundle = {cover_name: {"image/jpeg": base64.b64encode(cover).decode("ascii")}}
    diagram_bundle = {diagram_name: {"image/png": base64.b64encode(diagram).decode("ascii")}}
    anchor = lambda name: f"ai-{identity}-{name}"
    archival = entry["status"] == "archive"
    status = ("**Архивный учебный материал.** Код и сохранённые результаты относятся к исходному ноутбуку. "
              "Оформление не означает новый запуск или подтверждение старых метрик. Перед выполнением проверьте зависимости и ограничения.") if archival else (
              "**Стартовый пример на CPU.** Учебные данные создаются в коде. Проверка оформления не равна выполнению; "
              "результат запуска подтверждается отдельно.")
    intro = (f"![Обложка курса «{course}»](attachment:{cover_name})\n\n"
             f"AI PATH · {course} · Урок {identity}\n\n# {design['title']}\n\n{design['subtitle']}\n\n{status}")
    prep = (f"<a id=\"{anchor('goal')}\"></a>\n## Что вы разберёте\n\n{design['goal']}\n\n"
            "### Перед началом\n\n" + "\n".join(f"- {item}" for item in design["prerequisites"]) +
            "\n\n### Что понадобится\n\n" + "\n".join(f"- {item}" for item in design["setup"]))
    toc_items = [(anchor("goal"), "Цель и подготовка"), (anchor("map"), "Карта урока")]
    toc_items += [(anchor(f"section-{i:02d}"), step["title"]) for i, step in enumerate(design["steps"], 1)]
    toc_items += [(anchor("practice"), "Проверьте понимание"), (anchor("provenance"), "Источники и ограничения")]
    toc = "## Навигация по уроку\n\n" + "\n".join(f"- [{label}](#{target})" for target, label in toc_items)
    concept = (f"<a id=\"{anchor('map')}\"></a>\n## Карта урока\n\n"
               f"![{design['diagram']['title']}](attachment:{diagram_name})\n\n"
               "Схема показывает последовательность идей. Она не содержит результатов нового обучения или вычислений.\n\n" +
               "\n".join(f"{i}. **{node['label']}** — {node['detail']}"
                           for i, node in enumerate(design["diagram"]["nodes"], 1)))
    cells = [markdown(identity, "cover", intro, cover_bundle), markdown(identity, "preparation", prep),
             markdown(identity, "contents", toc), markdown(identity, "concept", concept, diagram_bundle)]
    sections = {step["beforeCell"]: (i, step) for i, step in enumerate(design["steps"], 1)}
    for index, cell in enumerate(notebook.cells):
        if index in sections:
            number, step = sections[index]
            heading = (f"<a id=\"{anchor(f'section-{number:02d}')}\"></a>\n"
                       f"## {number:02d} · {step['title']}\n\n{step['explanation']}\n\n"
                       "_Далее — исходные ячейки учебного материала._")
            cells.append(markdown(identity, f"section-{number:02d}", heading))
        cells.append(cell)
    practice = (f"<a id=\"{anchor('practice')}\"></a>\n## Проверьте понимание\n\n" +
                "\n".join(f"{i}. {item}" for i, item in enumerate(design["checks"], 1)) +
                f"\n\n### Один самостоятельный эксперимент\n\n{design['experiment']['task']}\n\n"
                f"**Критерий успеха:** {design['experiment']['successCriterion']}")
    limitations = "\n".join(f"- {item}" for item in design["limitations"])
    note = ("Исходные подписи, ссылки, авторство, код, изображения и аудио сохранены. Длинные текстовые выводы могут быть "
            "сокращены с указанием источника. Архивные результаты не пересчитывались.")
    if sanitization:
        note += " Раскрытые секреты и чужие настройки аккаунта заменены безопасными учебными примерами; исключения отмечены в аудите."
    provenance = (f"<a id=\"{anchor('provenance')}\"></a>\n## Источники и ограничения\n\n"
                  f"[Исходный ноутбук и его авторские материалы]({url}) · репозиторий `{manifest['repository']}`.\n\n"
                  f"{note}\n\n### Перед новым запуском\n\n{limitations}\n\n"
                  "**Иллюстрации:** обложка направления AI PATH; карта урока подготовлена по содержимому исходных ячеек.")
    cells.extend([markdown(identity, "practice", practice), markdown(identity, "provenance", provenance)])
    notebook.cells = cells
    notebook.nbformat, notebook.nbformat_minor = 4, 5
    attachments = [attachment_record(cover_name, "image/jpeg", cover),
                   attachment_record(diagram_name, "image/png", diagram)]
    metadata = {
        "schema": SCHEMA, "id": identity, "course": entry["course"], "status": entry["status"],
        "source": {"repository": manifest["repository"], "ref": manifest["ref"], "path": entry["path"],
                   "blob": entry.get("blob"), "url": url},
        "designSha256": digest(canonical(design)), "sourceCells": source_audit,
        "codeSha256": digest(canonical([r["sha256"] for r in code_audit])),
        "originalCodeSha256": digest(canonical([r["originalSha256"] for r in code_audit])),
        "originalCellCount": count, "originalCodeCellCount": len(code_audit),
        "reviewedCells": design["reviewedCells"], "sourceChangeExceptions": changed,
        "sanitization": sanitization, "textOutputCompaction": truncated,
        "originalMedia": existing_media, "attachments": attachments,
        "execution": "not-executed-by-presentation-builder",
    }
    notebook.metadata["ai_path_design"] = metadata
    try:
        nbformat.validate(notebook)
    except Exception:
        # Schema exceptions can include full cell payloads; do not expose them.
        raise ValueError(f"{identity}: styled notebook schema validation failed") from None
    report = {"id": identity, "path": entry["path"], "status": entry["status"],
              "originalCells": count, "styledCells": len(cells), "codeCells": len(code_audit),
              "codeSha256": metadata["codeSha256"], "originalCodeSha256": metadata["originalCodeSha256"],
              "sourceChangeExceptions": changed, "sanitization": sanitization,
              "compactedOutputs": len(truncated), "preservedMedia": len(existing_media),
              "preservedCellsSha256": digest(canonical([cell for cell in cells if cell.metadata.get("ai_path_original")])),
              "originalMetadataSha256": digest(canonical({key: value for key, value in notebook.metadata.items()
                                                           if key != "ai_path_design"})),
              "generatedMarkdownSha256": digest(canonical([{ "id": cell.id, "source": text(cell.source)}
                                                           for cell in cells if cell.metadata.get("ai_path_generated")])),
              "attachments": attachments, "validation": "PASS", "execution": "NOT_RUN"}
    return notebook, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--art-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--design", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve() if args.workspace else None
    repo = args.repo_root.resolve()
    manifest_path = args.manifest or ((workspace / "source" / "manifest.json") if workspace else
                                     repo / "tools" / "notebook-manifest.json")
    manifest = load_json(manifest_path)
    designs = load_designs(workspace, repo, args.design)
    entries = manifest["notebooks"]
    if len(entries) != 66 or {item["id"] for item in entries} != set(designs):
        raise ValueError("Require all 66 notebooks and exactly one reviewed design per notebook")
    if sum(item["status"] == "archive" for item in entries) != 65:
        raise ValueError("Expected 65 archives and one starter")
    covers = {course: packaged_cover(args.art_dir, course) for course in {e["course"] for e in entries}}
    reports, mapping = [], []
    for entry in entries:
        identity = entry["id"]
        baseline = read_baseline(entry, manifest, workspace, repo, args.source_dir)
        notebook, report = build_one(baseline, entry, designs[identity], manifest, covers[entry["course"]])
        destination = workspace / "styled" / f"{identity}.ipynb" if workspace else repo / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination, version=4)
        report["fileSha256"] = digest(destination.read_bytes())
        report["bytes"] = destination.stat().st_size
        reports.append(report)
        mapping.append({"id": identity, "localPath": str(destination.resolve()),
                        "repositoryPath": entry["path"], "sourceUrl": source_url(manifest, entry)})
        print(f"{identity}: styled; schema PASS; code preserved with {len(report['sourceChangeExceptions'])} documented exception(s)")
    review = workspace / "review" if workspace else repo / "reports" / "notebook-design"
    write_json(review / "build-results.json", {"schema": SCHEMA, "notebooks": reports,
               "total": len(reports), "archives": 65, "starters": 1, "legacyExecution": "NOT_RUN"})
    write_json(review / "local-notebook-map.json", mapping)
    print(f"BUILD PASS: {len(reports)} illustrated notebooks; no notebook code executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
