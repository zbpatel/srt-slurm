import ast
import re
import subprocess
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


def test_mmlu_pro_aa_v3_sampling_and_methodology() -> None:
    config = _load("mmlu_pro_aa_v3.eval-factory.yaml")
    params = config["config"]["params"]
    assert config["config"]["type"] == "mmlu_pro_aa_v3"
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["max_new_tokens"] == 65536
    assert params["parallelism"] == 128
    assert params["extra"]["n_samples"] == 1
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
    assert "tau2 structured tool-call and tool-result continuation gate passed" in text
    assert "error.code not in {404, 503}" in text
    assert 'tool_choice="required"' in text
    assert "SciCode local sandbox scored code-execution gate passed" in text
    assert 'recovery_source="$RECOVERY_ACCURACY_DIR/eval-results/scicode"' in text
    assert "Recovered {len(positions)} completed SciCode generations" in text


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
