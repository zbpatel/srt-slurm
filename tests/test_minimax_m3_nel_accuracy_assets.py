import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ASSET_DIR = Path(__file__).parents[1] / "configs" / "accuracy" / "minimax-m3"


def _load(name: str) -> dict:
    return yaml.safe_load((ASSET_DIR / name).read_text())


def _assert_minimax_reasoning(config: dict) -> None:
    endpoint = config["target"]["api_endpoint"]
    adapter = endpoint["adapter_config"]
    assert "use_reasoning" not in adapter
    assert "params_to_add" not in adapter
    interceptors = adapter["interceptors"]
    names = [item["name"] for item in interceptors]
    assert names.index("payload_modifier") < names.index("endpoint")
    payload_modifier = next(
        item["config"] for item in interceptors if item["name"] == "payload_modifier"
    )
    assert payload_modifier["params_to_add"] == {
        "chat_template_kwargs": {"thinking_mode": "enabled"}
    }
    reasoning = next(
        item["config"]
        for item in interceptors
        if item["name"] == "reasoning"
    )
    assert reasoning["start_reasoning_token"] == "<mm:think>"
    assert reasoning["end_reasoning_token"] == "</mm:think>"
    assert adapter["post_eval_hooks"] == [
        {
            "config": {
                "html_report_size": 5,
                "report_types": ["html", "json"],
            },
            "name": "post_eval_report",
        }
    ]
    assert "post_eval_hooks" not in endpoint
    assert endpoint["url"] == "__SRT_TARGET_URL__"


def test_mmmu_pro_sampling_and_multimodal_methodology() -> None:
    config = _load("ns_mmmu_pro.eval-factory.yaml")
    params = config["config"]["params"]
    assert config["config"]["type"] == "ns_mmmu_pro"
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["max_new_tokens"] == 65536
    assert params["parallelism"] == 128
    assert params["extra"]["num_repeats"] is None
    assert params["extra"]["use_sandbox"] is False
    assert params["extra"]["server_type"] == "vllm"
    assert params["extra"]["skip_data_dir_check"] is True
    _assert_minimax_reasoning(config)


def test_aa_lcr_sampling_repeats_and_authorized_judge() -> None:
    config = _load("ns_aa_lcr.eval-factory.yaml")
    params = config["config"]["params"]
    judge = params["extra"]["judge"]
    assert config["config"]["type"] == "ns_aa_lcr"
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["max_new_tokens"] == 65536
    assert params["parallelism"] == 64
    assert params["extra"]["num_repeats"] == 16
    assert judge["model_id"] == "nvidia/qwen/eccn-qwen-235b"
    assert judge["api_key"] == "INFERENCE_API_KEY"
    assert judge["parallelism"] == 64
    assert "nvapi-" not in (ASSET_DIR / "ns_aa_lcr.eval-factory.yaml").read_text()
    _assert_minimax_reasoning(config)


def test_gpqa_diamond_sampling_repeats_and_methodology() -> None:
    config = _load("gpqa_diamond_aa_v3.eval-factory.yaml")
    params = config["config"]["params"]
    assert config["config"]["type"] == "gpqa_diamond_aa_v3"
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["max_new_tokens"] == 65536
    assert params["parallelism"] == 128
    assert params["extra"]["n_samples"] == 16
    _assert_minimax_reasoning(config)


def test_tau2_telecom_sampling_trials_and_authorized_user_simulator() -> None:
    config = _load("tau2_bench_telecom.eval-factory.yaml")
    params = config["config"]["params"]
    user = params["extra"]["user"]
    assert config["config"]["type"] == "tau2_bench_telecom"
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["max_new_tokens"] == 65536
    assert params["parallelism"] == 64
    assert params["extra"]["n_samples"] == 3
    assert params["extra"]["max_steps"] == 100
    assert params["extra"]["judge"]["enabled"] is False
    assert user["model_id"] == "nvidia/qwen/eccn-qwen-235b"
    assert user["api_key"] == "INFERENCE_API_KEY"
    assert user["temperature"] == 0.0
    assert user["top_p"] == 1.0
    assert config["target"]["api_endpoint"]["stream"] is False
    _assert_minimax_reasoning(config)


def test_scicode_sampling_and_local_sandbox_contract() -> None:
    config = _load("ns_scicode.eval-factory.yaml")
    params = config["config"]["params"]
    assert config["config"]["type"] == "ns_scicode"
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["max_new_tokens"] == 65536
    assert params["parallelism"] == 16
    assert params["extra"]["use_sandbox"] is True
    assert params["extra"]["num_repeats"] is None
    assert params["extra"]["judge_support"] is False
    _assert_minimax_reasoning(config)


def test_nel_runner_is_valid_shell_and_does_not_enable_xtrace() -> None:
    script = ASSET_DIR / "run_nel_eval.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "set -x" not in script.read_text()
    text = script.read_text()
    assert "GPQA access requires HF_TOKEN_FILE" in text
    assert "tau2 auto tool-call and tool-result continuation diagnostic passed" in text
    assert "error.code not in {404, 503}" in text
    assert 'tool_choice="auto"' in text
    assert "proceeding to the full benchmark so this model behavior is scored" in text
    assert "tool-capability-first-response.json" in text
    assert "SciCode local sandbox scored code-execution gate passed" in text
    assert 'recovery_source="$RECOVERY_ACCURACY_DIR/eval-results/${recovery_dataset}"' in text
    assert "Recovered {len(positions)} completed NeMo Skills generations" in text
    assert "endpoint integrity gate passed with no NUL corruption" in text
    assert 'if [ "$task_name" = "ns_mmmu_pro" ]' in text
    assert "MMMU-Pro single-image and four-way concurrent-image gates passed" in text
    assert '"type": "image_url"' in text
    assert "ThreadPoolExecutor(max_workers=4)" in text
    assert "validate_nel_accuracy.py" in text


def test_accuracy_validator_is_valid_python_and_fail_closed() -> None:
    validator = ASSET_DIR / "validate_nel_accuracy.py"
    compile(validator.read_text(), str(validator), "exec")
    text = validator.read_text()
    assert "NUL byte found" in text
    assert ".jsonl-async" in text
    assert "EXPECTED_GENERATION_COUNT" in text
    assert "EXPECTED_SIMULATION_COUNT" in text
    assert "CANONICAL_METRIC" in text


def test_accuracy_validator_accepts_complete_tau2_native_outputs(tmp_path: Path) -> None:
    prefix = "run_telecom_llm_agent_model_user_simulator_model"
    simulations = [
        {"task_id": f"task-{task}", "trial": trial, "skipped": False}
        for trial in range(3)
        for task in range(114)
    ]
    (tmp_path / f"{prefix}.json").write_text(json.dumps({"simulations": simulations}))
    (tmp_path / f"{prefix}_results.json").write_text(
        json.dumps(
            {
                "num_tasks": 114,
                "num_trials": 3,
                "num_simulations": 342,
                "metrics": {"avg_reward": 0.9, "pass_at_k": {"1": 0.9}},
            }
        )
    )
    (tmp_path / f"{prefix}_termination_summary.json").write_text(
        json.dumps({"total_simulations": 342, "skipped_samples": 0})
    )
    (tmp_path / "results.yml").write_text(
        "results:\n  groups:\n    tau2_bench_telecom:\n      metrics:\n"
        "        pass@1:\n          scores:\n            pass@1:\n              value: 0.9\n"
        "        avg_reward:\n          scores:\n            avg_reward:\n              value: 0.9\n"
    )
    env = {
        **os.environ,
        "EXPECTED_SIMULATION_COUNT": "342",
        "EXPECTED_SCENARIO_COUNT": "114",
        "EVAL_REPEATS": "3",
        "CANONICAL_METRIC": "pass@1.pass@1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(ASSET_DIR / "validate_nel_accuracy.py"),
            str(tmp_path),
            "tau2_bench_telecom",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "accuracy validation passed" in completed.stdout


def _load_markup_repair():
    """Load the dependency-free repair function without importing vLLM."""
    source = (ASSET_DIR / "minimax_m3_tolerant_tool_parser.py").read_text()
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name == "re" for alias in node.names):
                keep.append(node)
        elif isinstance(node, ast.Assign):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "repair_elided_parameter_tags":
            keep.append(node)
    namespace: dict = {}
    exec(compile(ast.Module(keep, type_ignores=[]), str(ASSET_DIR), "exec"), namespace)
    return namespace["repair_elided_parameter_tags"]


def test_tolerant_minimax_parser_repairs_only_elided_tool_parameters() -> None:
    repair = _load_markup_repair()
    ns = "]<]minimax[>["
    broken = (
        "prefix"
        f"{ns}<tool_call>{ns}<invoke name=\"lookup_subscriber\">"
        f"{ns}test-123{ns}</account_id>{ns}</invoke>{ns}</tool_call>"
    )
    expected = broken.replace(
        f"{ns}test-123{ns}</account_id>",
        f"{ns}<account_id>test-123{ns}</account_id>",
    )
    assert repair(broken) == expected
    assert repair("ordinary answer") == "ordinary answer"


def test_dynamo_frontend_sitecustomize_registers_tolerant_parser() -> None:
    source = (ASSET_DIR / "sitecustomize.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "minimax_m3_tolerant_tool_parser" in imports


def test_tolerant_minimax_parser_preserves_tagged_and_repairs_mixed_parameters() -> None:
    repair = _load_markup_repair()
    ns = "]<]minimax[>["
    mixed = (
        f"{ns}<tool_call>{ns}<invoke name=\"f\">"
        f"{ns}7{ns}</count>"
        f"{ns}<label>ready{ns}</label>"
        f"{ns}</invoke>{ns}</tool_call>"
    )
    repaired = repair(mixed)
    assert f"{ns}<count>7{ns}</count>" in repaired
    assert f"{ns}<label>ready{ns}</label>" in repaired
