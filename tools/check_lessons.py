#!/usr/bin/env python3
"""Verify every lesson against its recipe; optionally execute in a fresh CPU kernel.

Each run has a separate temporary working directory. Only the new four-cell
lessons are executed. Archive notebooks and their commands are never executed.
"""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from PIL import Image

from build_lessons import build_notebook, code_sha256, load_manifest, require


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def output_sha256(notebook):
    payload = [{'execution_count': cell.execution_count, 'outputs': cell.outputs}
               for cell in notebook.cells if cell.cell_type == 'code']
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def inspect_notebook(actual, expected, entry, require_executed=False):
    nbformat.validate(actual)
    ident = entry['id']
    require(len(actual.cells) == len(expected.cells), f'{ident}: unexpected cells')
    assertions = 0
    images = 0
    for cell, reference in zip(actual.cells, expected.cells):
        for key in ('cell_type', 'source', 'id'):
            require(cell.get(key) == reference.get(key), f'{ident}/{reference.id}: {key} differs from recipe')
        for key, value in reference.metadata.items():
            require(cell.metadata.get(key) == value, f'{ident}/{reference.id}: metadata differs')
        require(cell.get('attachments', {}) == reference.get('attachments', {}), f'{ident}: altered attachment')
        for attachment in cell.get('attachments', {}).values():
            with Image.open(io.BytesIO(base64.b64decode(attachment['image/jpeg'], validate=True))) as image:
                image.load()
            images += 1
        if cell.cell_type == 'code':
            # The builder adds only this known IPython directive for visible plots.
            lines = cell.source.splitlines()
            require(all(not line.startswith(('%', '!')) or line == '%matplotlib inline' for line in lines),
                    f'{ident}: unexpected notebook command')
            source = '\n'.join(line for line in lines if line != '%matplotlib inline')
            tree = ast.parse(source, filename=f'{ident}/{cell.id}')
            count = sum(isinstance(node, ast.Assert) or
                        (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                         and isinstance(node.exc.func, ast.Name) and node.exc.func.id == 'AssertionError')
                        for node in ast.walk(tree))
            assertions += count
            if cell.metadata.ai_path_role == 'step-code':
                require(count > 0, f'{ident}/{cell.id}: missing executable check')
            require(not any(output.output_type == 'error' for output in cell.outputs), f'{ident}: saved error output')
            if require_executed:
                require(isinstance(cell.execution_count, int), f'{ident}/{cell.id}: not executed')
                if cell.metadata.ai_path_role == 'step-code':
                    require(bool(cell.outputs), f'{ident}/{cell.id}: no visible result')
    require(images == 4, f'{ident}: expected hero and three illustrated reactions')
    metadata = actual.metadata.get('ai_path_lesson', {})
    for key, value in expected.metadata.ai_path_lesson.items():
        if key != 'execution':
            require(metadata.get(key) == value, f'{ident}: stale {key}')
    if require_executed:
        execution = metadata.get('execution')
        require(isinstance(execution, dict) and execution.get('status') == 'passed', f'{ident}: no execution evidence')
        require(execution.get('codeSha256') == code_sha256(actual), f'{ident}: outputs belong to different code')
        require(execution.get('recipeSha256') == metadata['recipeSha256'], f'{ident}: stale executed recipe')
        require(execution.get('outputSha256') == output_sha256(actual), f'{ident}: changed saved outputs')
    return {'assertions': assertions, 'images': images, 'codeCells': 4,
            'codeSha256': code_sha256(actual), 'recipeSha256': metadata['recipeSha256']}


def configure_kernel(root):
    runtime = root / '.lesson-runtime'
    for key, directory in {
        'JUPYTER_CONFIG_DIR': runtime / 'config', 'JUPYTER_RUNTIME_DIR': runtime / 'jupyter',
        'IPYTHONDIR': runtime / 'ipython', 'MPLCONFIGDIR': runtime / 'matplotlib',
    }.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    kernels = runtime / 'data' / 'kernels' / 'python3'
    kernels.mkdir(parents=True, exist_ok=True)
    (kernels / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'AI PATH Python', 'language': 'python',
    }), encoding='utf-8')
    os.environ['JUPYTER_PATH'] = str(runtime / 'data')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--workspace', type=Path)
    mode.add_argument('--repo-root', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--require-executed', action='store_true')
    parser.add_argument('--ids', help='Comma-separated IDs, for CI shards or a bounded rerun')
    args = parser.parse_args()
    local = args.workspace is not None
    root = (args.workspace if local else args.repo_root).resolve()
    release = root / 'release' if local else root
    recipe_dir = root / 'recipes' if local else root / 'tools/lesson-recipes'
    manifest = load_manifest(release / 'tools/lesson-manifest.json')
    selected = set(args.ids.split(',')) if args.ids else {e['id'] for e in manifest['notebooks']}
    require(selected <= {e['id'] for e in manifest['notebooks']}, 'Unknown lesson ID')
    runtime = configure_kernel(root) if args.execute else None
    report_dir = root / ('review' if local else 'reports')
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'lesson-validation.json'
    report = {'schema': 1, 'startedAt': utc_now(), 'mode': 'execution' if args.execute else 'validation',
              'python': platform.python_version(), 'platform': platform.system(),
              'archiveRef': manifest['archiveRef'], 'selectedIds': sorted(selected),
              'packages': {name: importlib.metadata.version(name) for name in
                           ('numpy', 'matplotlib', 'Pillow', 'pandas', 'scikit-learn', 'nbformat', 'nbclient', 'ipykernel')},
              'lessons': [], 'status': 'in-progress', 'completedSelected': 0}
    # A bounded rerun retains only evidence whose code AND full recipe still match.
    prior = {}
    if args.ids and report_path.exists():
        prior = {item['id']: item for item in json.loads(report_path.read_text(encoding='utf-8')).get('lessons', [])}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    failures = []
    for entry in manifest['notebooks']:
        ident = entry['id']
        recipe = json.loads((recipe_dir / f'{ident}.json').read_text(encoding='utf-8'))
        expected = build_notebook(recipe, entry, manifest, release / 'assets/lesson-characters')
        if ident not in selected:
            old = prior.get(ident)
            if old and all(old.get(key) == expected.metadata.ai_path_lesson[key] for key in ('codeSha256', 'recipeSha256')):
                report['lessons'].append(old)
            continue
        path = root / 'styled' / f'{ident}.ipynb' if local else root / entry['path']
        started = time.monotonic()
        item = {'id': ident, 'title': recipe['title'], 'path': entry['path'], 'checkedAt': utc_now()}
        try:
            notebook = nbformat.read(path, as_version=4)
            item.update(inspect_notebook(notebook, expected, entry, args.require_executed and not args.execute))
            if args.execute:
                # Fresh sources, kernel, and working directory: saved results cannot mask a failure.
                notebook = expected
                with tempfile.TemporaryDirectory(prefix=f'lesson-{ident}-', dir=runtime) as working:
                    NotebookClient(notebook, timeout=90, startup_timeout=90, kernel_name='python3',
                                   allow_errors=False, record_timing=True,
                                   resources={'metadata': {'path': working}}).execute()
                notebook.metadata.ai_path_lesson.execution = {
                    'status': 'passed', 'at': utc_now(), 'python': platform.python_version(),
                    'codeSha256': item['codeSha256'], 'recipeSha256': item['recipeSha256'],
                    'outputSha256': output_sha256(notebook),
                    'durationSeconds': round(time.monotonic() - started, 3),
                }
                inspect_notebook(notebook, expected, entry, True)
                nbformat.write(notebook, path)
                item['execution'] = 'passed'
                item['outputSha256'] = output_sha256(notebook)
                item['outputCount'] = sum(len(c.outputs) for c in notebook.cells if c.cell_type == 'code')
            else:
                item['execution'] = 'verified-saved' if args.require_executed else 'not-run'
            item['status'] = 'passed'
        except Exception as error:
            item.update(status='failed', error=f'{type(error).__name__}: {error}')
            failures.append(ident)
        item['durationSeconds'] = round(time.monotonic() - started, 3)
        report['lessons'].append(item)
        report['updatedAt'] = utc_now()
        report['completedSelected'] += 1
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"{ident} {item['status'].upper()} {item['durationSeconds']:.1f}s", flush=True)
        if item['status'] == 'failed':
            print(item['error'], flush=True)
    report['finishedAt'] = utc_now()
    report['status'] = ('passed' if report['completedSelected'] == len(selected)
                        and all(item['status'] == 'passed' for item in report['lessons']) else 'failed')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(selected) - len(failures)}/{len(selected)} selected lessons passed. Report: {report_path}', flush=True)
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
