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


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot parse JSON artifact {path}: {error}")
    if not isinstance(value, dict):
        fail(f"expected a JSON object in {path}")
    return value


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
    # Tau2 writes its canonical aggregate metrics to a native JSON summary,
    # alongside the Eval Factory results.yml rather than metrics.json.
    candidates += sorted(root.glob("*_results.json"))
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
    if metric == "pass@1.pass@1":
        suffixes.update({"metrics.pass@1.scores.pass@1", "metrics.pass_at_k.1"})
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


def validate_telecom(root: Path, documents: list[tuple[Path, Any]]) -> None:
    expected = required_int("EXPECTED_SIMULATION_COUNT")
    scenarios = required_int("EXPECTED_SCENARIO_COUNT")
    trials = required_int("EVAL_REPEATS")
    if scenarios * trials != expected:
        fail("Telecom expected scenario/trial/simulation counts are inconsistent")

    result_paths = sorted(root.glob("*_results.json"))
    summary_paths = sorted(root.glob("*_termination_summary.json"))
    simulation_paths = sorted(
        path
        for path in root.glob("*_telecom_llm_agent_*_user_simulator_*.json")
        if not path.name.endswith(("_results.json", "_termination_summary.json"))
    )
    if len(result_paths) != 1 or len(summary_paths) != 1 or len(simulation_paths) != 1:
        fail(
            "Telecom must produce exactly one results, termination-summary, and "
            f"simulation JSON file; got {len(result_paths)}/{len(summary_paths)}/"
            f"{len(simulation_paths)}"
        )
    for path in (*result_paths, *summary_paths, *simulation_paths):
        scan_for_nul(path)

    results = load_json_object(result_paths[0])
    summary = load_json_object(summary_paths[0])
    simulation = load_json_object(simulation_paths[0])
    rows = simulation.get("simulations")
    if not isinstance(rows, list) or len(rows) != expected:
        actual = len(rows) if isinstance(rows, list) else "invalid"
        fail(f"Telecom simulation JSON has {actual} rows; expected {expected}")
    if results.get("num_tasks") != scenarios or results.get("num_trials") != trials:
        fail(
            "Telecom results cardinality is inconsistent: "
            f"got {results.get('num_tasks')} tasks x {results.get('num_trials')} trials; "
            f"expected {scenarios} x {trials}"
        )
    if results.get("num_simulations") != expected:
        fail(f"Telecom results report {results.get('num_simulations')} simulations; expected {expected}")
    if summary.get("total_simulations") != expected or summary.get("skipped_samples") != 0:
        fail(
            "Telecom termination summary is incomplete: "
            f"total={summary.get('total_simulations')}, skipped={summary.get('skipped_samples')}"
        )
    identities = {(row.get("task_id"), row.get("trial")) for row in rows if isinstance(row, dict)}
    if len(identities) != expected or any(
        not isinstance(row, dict) or row.get("skipped") for row in rows
    ):
        fail("Telecom simulations contain duplicate identities, malformed rows, or skipped episodes")

    pass_at_one = results.get("metrics", {}).get("pass_at_k", {}).get("1")
    avg_reward = results.get("metrics", {}).get("avg_reward")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (pass_at_one, avg_reward)
    ):
        fail("Telecom native results are missing numeric pass@1 or avg_reward")


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: validate_nel_accuracy.py <output-dir> <task-name>")
    root = Path(sys.argv[1])
    task = sys.argv[2]
    if not root.is_dir():
        fail(f"output directory does not exist: {root}")

    output_files = sorted(root.glob("**/output*.jsonl*"))
    if not output_files and task not in {"gpqa_diamond_aa_v3", "tau2_bench_telecom"}:
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
