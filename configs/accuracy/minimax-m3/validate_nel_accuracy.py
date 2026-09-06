#!/usr/bin/env python3
"""Fail closed when a MiniMax-M3 accuracy evaluation is incomplete or corrupt."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml


def fail(message: str) -> None:
    raise SystemExit(f"accuracy validation failed: {message}")


def required_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        fail(f"required environment variable {name} is missing")
    try:
        return int(value)
    except ValueError:
        fail(f"{name} is not an integer: {value!r}")


def scan_for_nul(path: Path) -> None:
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            if b"\x00" in chunk:
                fail(f"NUL byte found in {path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(f"invalid JSON in {path}:{line_number}: {error}")
            if not isinstance(row, dict):
                fail(f"non-object JSON row in {path}:{line_number}")
            rows.append(row)
    return rows


def require_count(path: Path, expected: int) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"expected output is missing: {path}")
    rows = load_jsonl(path)
    if len(rows) != expected:
        fail(f"{path} has {len(rows)} rows; expected {expected}")
    return rows


def flattened_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(key_path)
            keys.update(flattened_keys(child, key_path))
    elif isinstance(value, list):
        for child in value:
            keys.update(flattened_keys(child, prefix))
    return keys


def load_metric_documents(root: Path) -> list[tuple[Path, Any]]:
    candidates = sorted(root.glob("eval-results/**/metrics.json"))
    candidates += sorted(root.glob("**/eval_factory_metrics.json"))
    candidates += sorted(root.glob("**/results.yml"))
    documents: list[tuple[Path, Any]] = []
    for path in candidates:
        try:
            documents.append((path, yaml.safe_load(path.read_text())))
        except (OSError, yaml.YAMLError) as error:
            fail(f"cannot parse metric artifact {path}: {error}")
    if not documents:
        fail("no metrics.json, eval_factory_metrics.json, or results.yml was produced")
    return documents


def require_metric(documents: list[tuple[Path, Any]], metric: str) -> None:
    # NeMo Skills wraps metrics in a dataset name and represents avg-of-N as
    # pass@1 in metrics.json.  Accept only an exact requested suffix, plus that
    # documented normalization—not a nearby or substitute score.
    suffixes = {metric}
    suffixes.add(metric.replace("pass@1[avg-of-N]", "pass@1"))
    for _, document in documents:
        keys = flattened_keys(document)
        if any(
            key == suffix or key.endswith(f".{suffix}")
            for key in keys
            for suffix in suffixes
        ):
            return
    paths = ", ".join(str(path) for path, _ in documents)
    fail(f"canonical metric {metric!r} is absent from: {paths}")


def validate_simple_evals(root: Path) -> None:
    expected = required_int("EXPECTED_GENERATION_COUNT")
    databases = sorted(root.glob("**/cache/cache.sqlite/cache.db"))
    if len(databases) != 1:
        fail(f"expected exactly one simple-evals cache database, found {len(databases)}")
    database = databases[0]
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        count = connection.execute("SELECT COUNT(*) FROM Cache").fetchone()[0]
    finally:
        connection.close()
    if quick_check != "ok":
        fail(f"simple-evals cache quick_check returned {quick_check!r}")
    if count != expected:
        fail(f"simple-evals cache has {count} rows; expected {expected}")


def validate_aa_lcr(root: Path) -> None:
    questions = required_int("EXPECTED_QUESTION_COUNT")
    repeats = required_int("EVAL_REPEATS")
    expected_total = required_int("EXPECTED_GENERATION_COUNT")
    if questions * repeats != expected_total:
        fail("AA-LCR expected question/repeat/generation counts are inconsistent")
    for base in (root / "tmp-eval-results" / "aalcr", root / "eval-results" / "aalcr"):
        expected_names = {f"output-rs{seed}.jsonl" for seed in range(repeats)}
        actual = {path.name for path in base.glob("output-rs*.jsonl")}
        if actual != expected_names:
            fail(f"{base} has final files {sorted(actual)}; expected {sorted(expected_names)}")
        total = sum(len(require_count(base / name, questions)) for name in sorted(expected_names))
        if total != expected_total:
            fail(f"{base} has {total} rows; expected {expected_total}")


def validate_mmmu(root: Path) -> None:
    expected = required_int("EXPECTED_GENERATION_COUNT")
    require_count(root / "eval-results" / "mmmu-pro" / "output.jsonl", expected)


def validate_scicode(root: Path, documents: list[tuple[Path, Any]]) -> None:
    problems = required_int("EXPECTED_PROBLEM_COUNT")
    subtasks = required_int("EXPECTED_SUBTASK_COUNT")
    require_count(root / "eval-results" / "scicode" / "output.jsonl", problems)
    metrics = next(
        (
            document.get("scicode", {}).get("pass@1", {})
            for path, document in documents
            if path.name == "metrics.json" and isinstance(document, dict)
        ),
        {},
    )
    if metrics.get("num_problems") != problems or metrics.get("num_subtasks") != subtasks:
        fail(
            "SciCode metrics cardinality is incomplete: "
            f"got {metrics.get('num_problems')}/{metrics.get('num_subtasks')}, "
            f"expected {problems}/{subtasks}"
        )


def find_numeric_key(value: Any, names: set[str]) -> list[float]:
    found: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in names and isinstance(child, (int, float)) and not isinstance(child, bool):
                found.append(float(child))
            found.extend(find_numeric_key(child, names))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_numeric_key(child, names))
    return found


def validate_telecom(root: Path, documents: list[tuple[Path, Any]]) -> None:
    expected = required_int("EXPECTED_SIMULATION_COUNT")
    count_keys = {"num_entries", "num_episodes", "num_samples", "num_simulations"}
    metric_counts = [
        number
        for _, document in documents
        for number in find_numeric_key(document, count_keys)
    ]
    if expected in metric_counts:
        return
    outputs = [
        path
        for path in root.glob("eval-results/**/*.jsonl")
        if path.name.startswith("output")
    ]
    row_counts = [len(load_jsonl(path)) for path in outputs]
    if expected not in row_counts and sum(row_counts) != expected:
        fail(
            f"Telecom produced neither a metric nor final JSONL cardinality of {expected}; "
            f"metric counts={metric_counts}, output counts={row_counts}"
        )


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: validate_nel_accuracy.py <output-dir> <task-name>")
    root = Path(sys.argv[1])
    task = sys.argv[2]
    if not root.is_dir():
        fail(f"output directory does not exist: {root}")

    output_files = sorted(root.glob("**/output*.jsonl*"))
    if not output_files and task != "gpqa_diamond_aa_v3":
        fail("no evaluator JSONL output files were produced")
    for path in output_files:
        scan_for_nul(path)
    incomplete = [path for path in output_files if path.name.endswith(".jsonl-async") and path.stat().st_size]
    if incomplete:
        fail("non-empty incomplete async outputs remain: " + ", ".join(map(str, incomplete)))

    documents = load_metric_documents(root)
    metrics = os.environ.get("CANONICAL_METRICS") or os.environ.get("CANONICAL_METRIC")
    if not metrics:
        fail("CANONICAL_METRIC or CANONICAL_METRICS is missing")
    for metric in metrics.split(","):
        require_metric(documents, metric.strip())

    if task == "gpqa_diamond_aa_v3":
        validate_simple_evals(root)
    elif task == "ns_aa_lcr":
        validate_aa_lcr(root)
    elif task == "ns_mmmu_pro":
        validate_mmmu(root)
    elif task == "ns_scicode":
        validate_scicode(root, documents)
    elif task == "tau2_bench_telecom":
        validate_telecom(root, documents)
    else:
        fail(f"unsupported MiniMax-M3 accuracy task: {task}")

    print(f"accuracy validation passed for {task}: complete cardinality, metrics, and JSON integrity")


if __name__ == "__main__":
    main()
