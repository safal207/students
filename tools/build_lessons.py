#!/usr/bin/env python3
"""Build portable AI PATH notebooks from reviewed lesson recipes; never execute code.

Local:  python build_lessons.py --workspace /path/to/work/story-lessons
Repo:   python tools/build_lessons.py --repo-root .

build_notebook() returns a fresh notebook without outputs. Validators can compare
its sources with an executed notebook without discarding saved execution results.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
import struct
from urllib.parse import quote, urlsplit

import nbformat


ARTS = {"start", "think", "check", "fix", "compare", "finish"}
FINDING_LABELS = {"bug": "Ошибка", "teaching": "Понятнее для новичка",
                  "limitation": "Ограничение исходного примера"}
REQUIRED = {"id", "title", "subtitle", "goal", "scope", "prerequisites",
            "dataDescription", "setupCode", "steps", "experiment", "takeaways",
            "limitations", "findings", "dependencies", "reviewedOriginalCells",
            "sourceAttributions"}
PLOT_STYLE = """%matplotlib inline
import matplotlib as mpl
mpl.rcParams.update({
    'figure.figsize': (7, 3.5), 'figure.dpi': 100,
    'figure.facecolor': '#fffdf8', 'axes.facecolor': '#fffdf8',
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelcolor': '#302c3f',
    'text.color': '#302c3f', 'savefig.bbox': 'tight',
})
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: expected nonempty text")
    return value.strip()


def text_list(value, label, minimum=0):
    require(isinstance(value, list) and len(value) >= minimum, f"{label}: expected list")
    for item in value:
        text(item, label)


def valid_url(value):
    parts = urlsplit(text(value, "URL"))
    require(parts.scheme in {"https", "http"} and bool(parts.netloc)
            and not any(char.isspace() for char in value), f"Unsupported URL: {value}")


def safe_relative_path(value):
    require(isinstance(value, str) and "\\" not in value, "Manifest paths must use forward slashes")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.parts and ".." not in path.parts,
            f"Path must stay inside the repository: {value}")
    require(path.suffix == ".ipynb", f"Not a notebook path: {value}")
    return path


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    require(re.fullmatch(r"[\w.-]+/[\w.-]+", manifest.get("repository", "")), "Invalid repository")
    require(re.fullmatch(r"[0-9a-f]{40}", manifest.get("archiveRef", "")),
            "archiveRef must be a complete immutable commit SHA")
    entries = manifest.get("notebooks")
    require(isinstance(entries, list), "Manifest notebooks must be a list")
    require(sorted(item.get("id", "") for item in entries) == [f"{n:03}" for n in range(1, 67)],
            "Manifest must contain every lesson ID 001–066 exactly once")
    paths = [str(safe_relative_path(item["path"])) for item in entries]
    require(len(paths) == len(set(paths)), "Manifest contains duplicate notebook paths")
    for entry in entries:
        text(entry.get("course"), "course")
        text(entry.get("title"), "title")
        require(entry.get("status") == ("starter" if entry["id"] == "018" else "lesson"),
                f"{entry['id']}: invalid lesson status")
        require(type(entry.get("originalCells")) is int and entry["originalCells"] > 0,
                f"{entry['id']}: originalCells must be a positive integer")
    return manifest


def validate_recipe(recipe, entry):
    """Validate all authoring fields before any output notebook is written."""
    require(isinstance(recipe, dict) and REQUIRED <= recipe.keys(), "Missing recipe fields")
    ident = entry["id"]
    require(recipe["id"] == ident, f"Recipe ID differs from manifest: {ident}")
    require(recipe["title"] == entry["title"], f"{ident}: title differs from manifest")
    for field in ("title", "subtitle", "goal", "scope", "dataDescription"):
        text(recipe[field], f"{ident}.{field}")
    require(isinstance(recipe["setupCode"], str), f"{ident}.setupCode: expected a string")
    for field, minimum in (("prerequisites", 1), ("takeaways", 2), ("limitations", 1),
                           ("dependencies", 0), ("sourceAttributions", 0)):
        text_list(recipe[field], f"{ident}.{field}", minimum)
    require(len(recipe["prerequisites"]) <= 3, f"{ident}: at most three prerequisites")
    reviewed = recipe["reviewedOriginalCells"]
    require(isinstance(reviewed, list) and all(type(index) is int for index in reviewed)
            and reviewed == list(range(entry["originalCells"])), f"{ident}: incomplete original-cell review")
    steps = recipe["steps"]
    require(isinstance(steps, list) and len(steps) == 3, f"{ident}: exactly three steps required")
    for number, step in enumerate(steps, 1):
        require(isinstance(step, dict), f"{ident}: step {number} must be an object")
        for field in ("title", "explanation", "code", "expected"):
            text(step.get(field), f"{ident}.step{number}.{field}")
        reaction = step.get("reaction")
        require(isinstance(reaction, dict) and reaction.get("art") in ARTS,
                f"{ident}: invalid reaction art in step {number}")
        for actor in ("learner", "mentor", "bot"):
            text(reaction.get(actor), f"{ident}.step{number}.{actor}")
    require(isinstance(recipe["experiment"], dict), f"{ident}: experiment must be an object")
    for field in ("task", "hint", "success"):
        text(recipe["experiment"].get(field), f"{ident}.experiment.{field}")
    require(isinstance(recipe["findings"], list) and recipe["findings"], f"{ident}: findings required")
    for finding in recipe["findings"]:
        require(isinstance(finding, dict) and finding.get("kind") in FINDING_LABELS,
                f"{ident}: invalid finding kind")
        index = finding.get("originalCell")
        require(type(index) is int and index in reviewed, f"{ident}: finding references an unknown cell")
        for field in ("problem", "evidence", "resolution", "check"):
            text(finding.get(field), f"{ident}.finding.{field}")
    for url in recipe["sourceAttributions"]:
        valid_url(url)
    compile(recipe["setupCode"] + "\n" + "\n".join(step["code"] for step in steps), ident, "exec")


def canonical_recipe_sha256(recipe):
    payload = json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_sha256(notebook):
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    return hashlib.sha256("\n\n".join(sources).encode("utf-8")).hexdigest()


def jpeg_attachment(path, max_width, max_height):
    """Check JPEG dimensions using its SOF header, without an imaging dependency."""
    data = path.read_bytes()
    require(data.startswith(b"\xff\xd8"), f"Not a JPEG: {path}")
    position = 2
    size = None
    while position < len(data):
        require(data[position] == 255, f"Invalid JPEG marker: {path}")
        while position < len(data) and data[position] == 255:
            position += 1
        require(position < len(data), f"Truncated JPEG: {path}")
        marker = data[position]
        position += 1
        if marker in {0xD9, 0xDA}:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        require(position + 2 <= len(data), f"Truncated JPEG segment: {path}")
        length = struct.unpack_from(">H", data, position)[0]
        require(length >= 2 and position + length <= len(data), f"Invalid JPEG segment: {path}")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            require(length >= 8, f"Invalid JPEG dimensions: {path}")
            height, width = struct.unpack_from(">HH", data, position + 3)
            size = (width, height)
            break
        position += length
    require(size is not None and 0 < size[0] <= max_width and 0 < size[1] <= max_height,
            f"JPEG must fit {max_width}×{max_height}: {path} ({size})")
    return {"image/jpeg": base64.b64encode(data).decode("ascii")}


def build_notebook(recipe, entry, manifest, image_dir):
    """Build a deterministic, unexecuted notebook; all figures are attachments."""
    validate_recipe(recipe, entry)
    ident = recipe["id"]
    archive = (f"https://github.com/{manifest['repository']}/blob/"
               f"{manifest['archiveRef']}/{quote(entry['path'], safe='/')}")
    images = Path(image_dir)
    cells = []

    def markdown(role, source, suffix=None, art=None, hero=False, step=None):
        cell = nbformat.v4.new_markdown_cell(source.strip(), id=f"lesson-{ident}-{suffix or role}")
        cell.metadata["ai_path_role"] = role
        if step is not None:
            cell.metadata["ai_path_step"] = step
        if art:
            filename = "start-hero.jpg" if hero else f"{art}.jpg"
            limits = (960, 640) if hero else (480, 320)
            cell.attachments = {filename: jpeg_attachment(images / filename, *limits)}
            cell.metadata["ai_path_art"] = art
        cells.append(cell)

    def code(role, source, suffix=None, step=None):
        cell = nbformat.v4.new_code_cell(source.strip() + "\n", id=f"lesson-{ident}-{suffix or role}")
        cell.metadata["ai_path_role"] = role
        if step is not None:
            cell.metadata["ai_path_step"] = step
        cells.append(cell)

    markdown("intro", f"# {recipe['title']}\n\n"
             "![Мира, Данил и Пикс — команда AI PATH](attachment:start-hero.jpg)\n\n"
             f"**{recipe['subtitle']}**\n\n**Твой результат:** {recipe['goal']}", art="start", hero=True)
    prerequisites = "\n".join(f"- {item}" for item in recipe["prerequisites"])
    # GitHub's notebook iframe rewrites fragment-only links to a repository folder.
    # A plain numbered route works in GitHub, Jupyter and Colab without a false jump.
    contents = "\n".join(f"{number}. {step['title']}"
                         for number, step in enumerate(recipe["steps"], 1))
    markdown("overview", f"**Что разбираем:** {recipe['scope']}\n\n"
             f"**Данные:** {recipe['dataDescription']}\n\n**Перед началом:**\n\n{prerequisites}\n\n"
             "**Рядом с тобой:** Мира — ученица, Данил — человек-ментор, Пикс — бот-помощник. "
             "Это вымышленные персонажи: их реплики объясняют шаги и не проверяют код автоматически.\n\n"
             f"## Три шага\n\n{contents}\n\nЗатем — твой небольшой эксперимент. Проходи ячейки сверху вниз.")
    dependencies = ", ".join(recipe["dependencies"]) or "только стандартный Python"
    markdown("setup-guide", "## Подготовь рабочее место\n\n"
             "Запускай код сверху вниз: сначала эту подготовку, затем три шага. "
             "Проверки `assert` останавливают выполнение, если результат не совпал с условием.\n\n"
             f"Библиотеки для урока: **{dependencies}**.")
    setup = (PLOT_STYLE if "matplotlib" in recipe["dependencies"] else "") + recipe["setupCode"]
    if not ast.parse(recipe['setupCode']).body and 'matplotlib' not in recipe['dependencies']:
        setup = "print('Рабочее место готово: используем встроенный Python.')"
    code("setup", setup)
    for number, step in enumerate(recipe["steps"], 1):
        markdown("step", f'<a id="step-{number}"></a>\n\n## {number}. {step["title"]}\n\n'
                 f'{step["explanation"]}', suffix=f"step-{number}", step=number)
        code("step-code", step["code"], suffix=f"step-{number}-code", step=number)
        reaction = step["reaction"]
        dialogue = "\n\n".join(f"> **{name}:** " + reaction[key].replace("\n", "\n> ")
                                  for name, key in (("Мира", "learner"), ("Данил", "mentor"), ("Пикс", "bot")))
        markdown("reaction", f"**Что проверить:** {step['expected']}\n\n"
                 f"![Команда AI PATH обсуждает шаг {number}](attachment:{reaction['art']}.jpg)\n\n{dialogue}",
                 suffix=f"reaction-{number}", art=reaction["art"], step=number)
    experiment = recipe["experiment"]
    markdown("experiment", '<a id="your-experiment"></a>\n\n## Попробуй сам\n\n'
             f"{experiment['task']}\n\n**Где изменить:** {experiment['hint']}\n\n"
             f"**Как проверить:** {experiment['success']}")
    markdown("takeaways", "## Что теперь умеешь\n\n" + "\n".join(f"- {item}" for item in recipe["takeaways"]))
    markdown("limitations", "## Границы этого примера\n\n" + "\n".join(f"- {item}" for item in recipe["limitations"]))
    sources = "\n".join(f"- [Источник {index}](<{url}>)"
                        for index, url in enumerate(recipe["sourceAttributions"], 1))
    markdown("archive", "## Продолжить глубже\n\n"
             f"[Полная исходная работа в неизменяемом архиве](<{archive}>). "
             "Она сохраняет большой эксперимент и его авторские материалы; её требования могут отличаться от этого короткого урока."
             + (f"\n\n**Источники исходной работы:**\n\n{sources}" if sources else ""))
    findings = []
    for finding in recipe["findings"]:
        findings.append(f"**{FINDING_LABELS[finding['kind']]} · исходная ячейка {finding['originalCell']} (от 0).** "
                        f"{finding['problem']} → {finding['resolution']}\n\n"
                        f"Проверка: {finding['check']}\n\n"
                        f"Фрагмент исходника: <code>{html.escape(finding['evidence'])}</code>")
    markdown("findings", "<details>\n<summary>Что исправлено и упрощено в уроке</summary>\n\n"
             + "\n\n---\n\n".join(findings) + "\n\n</details>")
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"},
        "ai_path_lesson": {"schema": 1, "id": ident, "course": entry["course"],
                           "status": entry["status"], "recipeSha256": canonical_recipe_sha256(recipe),
                           "archiveUrl": archive, "execution": "not-run"},
    })
    notebook.metadata.ai_path_lesson["codeSha256"] = code_sha256(notebook)
    nbformat.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--workspace", type=Path)
    mode.add_argument("--repo-root", type=Path)
    parser.add_argument("--ids", help="Optional comma-separated lesson IDs to rebuild")
    args = parser.parse_args()
    local = args.workspace is not None
    root = (args.workspace if local else args.repo_root).resolve()
    release = root / "release" if local else root
    manifest = load_manifest(release / "tools" / "lesson-manifest.json")
    selected = set(args.ids.split(',')) if args.ids else {entry['id'] for entry in manifest['notebooks']}
    require(selected <= {entry['id'] for entry in manifest['notebooks']}, 'Unknown lesson ID')
    recipes = root / "recipes" if local else release / "tools" / "lesson-recipes"
    image_dir = release / "assets" / "lesson-characters"
    # Prepare every notebook first: a missing image or invalid recipe cannot leave a partial build.
    pending = []
    for entry in sorted(manifest["notebooks"], key=lambda item: item["id"]):
        if entry['id'] not in selected:
            continue
        recipe = json.loads((recipes / f"{entry['id']}.json").read_text(encoding="utf-8"))
        notebook = build_notebook(recipe, entry, manifest, image_dir)
        destination = root / "styled" / f"{entry['id']}.ipynb" if local else root / entry["path"]
        require(destination.resolve().is_relative_to(root), f"Output escapes root: {destination}")
        pending.append((entry, notebook, destination))
    report = {"schema": 1, "execution": "not-run", "archiveRef": manifest["archiveRef"], "lessons": []}
    for entry, notebook, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
        report["lessons"].append({"id": entry["id"], "path": destination.relative_to(root).as_posix(),
                                  "cells": len(notebook.cells),
                                  "codeLines": sum(len(cell.source.splitlines()) for cell in notebook.cells if cell.cell_type == "code"),
                                  "images": sum(len(cell.get("attachments", {})) for cell in notebook.cells),
                                  "codeSha256": code_sha256(notebook)})
    report_path = root / ("review" if local else "reports") / "lesson-build.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(pending)} notebooks; code execution: not run. Report: {report_path}")


if __name__ == "__main__":
    main()
