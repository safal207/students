"""Narrow, documented removal of exposed credentials; never execute source code."""
from __future__ import annotations

import ast
import copy
import re


def _source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _replace_strings(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            if old:
                value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        for key in list(value):
            value[key] = _replace_strings(value[key], replacements)
    return value


def sanitize_notebook(nb, identity):
    """Return an independent notebook and non-sensitive change descriptions.

    Removing a credential from this revision does not revoke it or erase history.
    Existing figure/audio payloads are left intact. Only known source locations
    are changed, and literal replacements are restricted to the affected file.
    """
    nb = copy.deepcopy(nb)
    notes = []
    identity = str(identity).zfill(3)
    if identity == "064":
        cell = nb.cells[18]
        source = _source(cell)
        safe_wifi_source = 'stroka = "Учебный пример: пароль Wi-Fi — DEMO_ONLY_123."'
        if source != safe_wifi_source:
            try:
                assignment = ast.parse(source).body[0]
                old = ast.literal_eval(assignment.value)
                assert isinstance(assignment, ast.Assign)
                assert assignment.targets[0].id == "stroka" and isinstance(old, str)
            except (ValueError, SyntaxError, AttributeError, AssertionError, IndexError):
                raise ValueError("064:18 unexpected source structure; manual review required") from None
            replacement = "Учебный пример: пароль Wi-Fi — DEMO_ONLY_123."
            _replace_strings(nb, [(old, replacement)])
            cell.source = safe_wifi_source
        notes.append({"cell": 18, "reason": "replaced potentially private Wi-Fi sample with synthetic training text"})
    if identity == "066":
        # Old Colab exported stream tags in an output field disallowed by nbformat.
        # Preserve that metadata at cell level; do not discard stream text or media.
        legacy_stream_metadata = {}
        for output_index, output in enumerate(nb.cells[18].get("outputs", [])):
            if output.get("output_type") == "stream" and "metadata" in output:
                legacy_stream_metadata[str(output_index)] = output.pop("metadata")
        if legacy_stream_metadata:
            nb.cells[18].metadata["ai_path_preserved_stream_metadata"] = legacy_stream_metadata
        if nb.cells[18].metadata.get("ai_path_preserved_stream_metadata"):
            notes.append({"cell": 18, "reason": "moved legacy stream metadata to cell metadata for nbformat validity; output text preserved"})
        cell = nb.cells[31]
        source = _source(cell)
        safe_deployment_source = (
            "import os\n\n"
            "# Задайте новый токен в окружении или хранилище секретов.\n"
            "# Не вставляйте токен в ноутбук. Старый раскрытый токен нужно отозвать.\n"
            "# Heroku CLI читает HEROKU_API_KEY из окружения:\n"
            "# https://devcenter.heroku.com/articles/authentication\n"
            'if not os.environ.get("HEROKU_API_KEY"):\n'
            '    raise RuntimeError("Задайте HEROKU_API_KEY безопасно вне ноутбука перед ручным развёртыванием.")'
        )
        if source != safe_deployment_source:
            literals = re.findall(r"(?m)^\s*(API_KEY|LOGIN)\s*=\s*(['\"])(.*?)\2", source)
            if {name for name, _, _ in literals} != {"API_KEY", "LOGIN"} or ".netrc" not in source:
                raise ValueError("066:31 unexpected source structure; manual review required")
            _replace_strings(nb, [(value, "[REMOVED_CREDENTIAL]") for _, _, value in literals if value])
            cell.source = safe_deployment_source
            cell.outputs = []
            cell.execution_count = None
        identity_cell = nb.cells[27]
        identity_cell.source = (
            "# При необходимости настройте автора только в своём репозитории:\n"
            '# !git config user.email "you@example.com"\n'
            '# !git config user.name "Your Name"'
        )
        identity_cell.outputs = []
        identity_cell.execution_count = None
        notes.extend([
            {"cell": 27, "reason": "replaced another author's global Git identity with commented local examples"},
            {"cell": 31, "reason": "removed hardcoded deployment credentials and netrc write; environment lookup only", "revocation": "owner action required; history is unchanged"},
        ])
    return nb, notes
