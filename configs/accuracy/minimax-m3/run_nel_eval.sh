#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <eval-factory-template.yaml> <task-name>" >&2
    exit 2
fi

template=$1
task_name=$2
target_url="http://${SRT_FRONTEND_HOST:?}:${SRT_FRONTEND_PORT:?}/v1/chat/completions"
output_dir="/logs/accuracy/${task_name}"
resolved_config="${output_dir}/config_ef.resolved.yaml"

if [ ! -r "$template" ]; then
    echo "NEL eval-factory template is not readable: $template" >&2
    exit 2
fi

mkdir -p "$output_dir"

# A Slurm time-limit recovery may seed only the evaluator's completed-work
# cache into a new submission-scoped output directory.  Both harnesses are
# explicitly resume-safe: simple-evals reads its response cache, while NeMo
# Skills launches generation with skip_filled=True and retains async position
# IDs until the final ordered file is complete.
if [ -n "${RECOVERY_ACCURACY_DIR:-}" ]; then
    if [ ! -d "$RECOVERY_ACCURACY_DIR" ]; then
        echo "Recovery accuracy directory does not exist: $RECOVERY_ACCURACY_DIR" >&2
        exit 2
    fi

    case "$task_name" in
        mmlu_pro_aa_v3|gpqa_diamond_aa_v3)
            recovery_source="$RECOVERY_ACCURACY_DIR/${task_name}/cache"
            recovery_target="$output_dir/${task_name}/cache"
            ;;
        ns_aa_lcr)
            recovery_source="$RECOVERY_ACCURACY_DIR/tmp-eval-results/aalcr"
            recovery_target="$output_dir/tmp-eval-results/aalcr"
            ;;
        ns_scicode)
            recovery_source="$RECOVERY_ACCURACY_DIR/eval-results/scicode"
            recovery_target="$output_dir/eval-results/scicode"
            ;;
        *)
            echo "Recovery is not configured for task: $task_name" >&2
            exit 2
            ;;
    esac

    if [ ! -d "$recovery_source" ]; then
        echo "Recovery source does not exist: $recovery_source" >&2
        exit 2
    fi
    mkdir -p "$recovery_target"
    cp -a "$recovery_source/." "$recovery_target/"

    if [ "$task_name" = "mmlu_pro_aa_v3" ] || [ "$task_name" = "gpqa_diamond_aa_v3" ]; then
        python3 - "$recovery_target/cache.sqlite/cache.db" <<'PY'
import sqlite3
import sys

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
result = db.execute("PRAGMA quick_check").fetchone()[0]
db.close()
if result != "ok":
    raise SystemExit(f"Recovered simple-evals cache failed SQLite quick_check: {result}")
print("Recovered simple-evals cache passed SQLite quick_check")
PY
    elif [ "$task_name" = "ns_aa_lcr" ]; then
        python3 - "$recovery_target" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
total = 0
for path in sorted(root.glob("output-rs*.jsonl-async")):
    positions = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            positions.append(json.loads(line)["_async_position"])
    if len(positions) != len(set(positions)):
        raise SystemExit(f"Duplicate async positions in recovered file: {path.name}")
    total += len(positions)
if total == 0:
    raise SystemExit("Recovered AA-LCR output contains no completed generations")
print(f"Recovered {total} completed AA-LCR generations with unique async positions")
PY
    else
        python3 - "$recovery_target/output.jsonl-async" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
positions = []
with path.open(encoding="utf-8") as stream:
    for line in stream:
        positions.append(json.loads(line)["_async_position"])
if not positions:
    raise SystemExit("Recovered SciCode output contains no completed generations")
if len(positions) != len(set(positions)):
    raise SystemExit("Duplicate async positions in recovered SciCode output")
print(f"Recovered {len(positions)} completed SciCode generations with unique async positions")
PY
    fi
fi

sed \
    -e "s|__SRT_TARGET_URL__|${target_url}|g" \
    -e "s|__NEL_OUTPUT_DIR__|${output_dir}|g" \
    "$template" > "$resolved_config"
cp "$template" "${output_dir}/config_ef.template.yaml"

echo "NEL task: ${task_name}"
echo "Target: ${target_url}"
echo "Resolved config: ${resolved_config}"

# GPQA is gated on Hugging Face. Read its credential from an ephemeral file
# mounted by AIB so the value never enters a recipe, srtctl's environment
# summary, the GitLab trace, or collected result artifacts.
if [ "$task_name" = "gpqa_diamond_aa_v3" ]; then
    : "${HF_TOKEN_FILE:?GPQA access requires HF_TOKEN_FILE}"
    if [ ! -r "$HF_TOKEN_FILE" ]; then
        echo "Hugging Face token file is not readable" >&2
        exit 2
    fi
    HF_TOKEN=$(<"$HF_TOKEN_FILE")
    export HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo "Hugging Face token file is empty" >&2
        exit 2
    fi
fi

# AA-LCR uses an external judge and tau2 Telecom uses the same authorized
# Qwen-235B endpoint as its user simulator. Read the credential from the
# ephemeral GitLab Secure File staged by AIB; never place it in the recipe,
# command line, stdout, or result artifacts. A real chat request catches
# authorization and model-name errors before the expensive phase begins.
if [ "$task_name" = "ns_aa_lcr" ] || [ "$task_name" = "tau2_bench_telecom" ]; then
    : "${JUDGE_API_KEY_FILE:?External Qwen-235B access requires JUDGE_API_KEY_FILE}"
    if [ ! -r "$JUDGE_API_KEY_FILE" ]; then
        echo "External API key file is not readable" >&2
        exit 2
    fi
    INFERENCE_API_KEY=$(<"$JUDGE_API_KEY_FILE")
    export INFERENCE_API_KEY
    if [ -z "$INFERENCE_API_KEY" ]; then
        echo "External API key file is empty" >&2
        exit 2
    fi

    judge_probe="${output_dir}/judge-probe.json"
    judge_status=$(curl --silent --show-error \
        --output "$judge_probe" \
        --write-out '%{http_code}' \
        --connect-timeout 30 \
        --max-time 120 \
        --header "Authorization: Bearer ${INFERENCE_API_KEY}" \
        --header 'Content-Type: application/json' \
        --data '{"model":"nvidia/qwen/eccn-qwen-235b","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":2,"temperature":0}' \
        'https://inference-api.nvidia.com/v1/chat/completions')
    if [ "$judge_status" != "200" ]; then
        echo "Qwen-235B authorization probe failed with HTTP ${judge_status}" >&2
        exit 1
    fi
    echo "Qwen-235B authorization probe passed"
fi

# tau2 requires a functioning OpenAI tool-call round trip, not merely an HTTP
# health check. Require a structured tool call and then a successful
# tool-result continuation from the exact endpoint before starting the full
# 114-task x3-trial evaluation.
if [ "$task_name" = "tau2_bench_telecom" ]; then
    python3 - "$target_url" "${output_dir}/tool-capability-first-response.json" <<'PY'
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

url = sys.argv[1]
capture_path = pathlib.Path(sys.argv[2])
tool = {
    "type": "function",
    "function": {
        "name": "lookup_subscriber",
        "description": "Look up a telecom subscriber by account ID.",
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
}
user_message = {
    "role": "user",
    "content": "Use lookup_subscriber for account test-123 before answering.",
}


def complete(messages, *, tools=None, tool_choice=None):
    payload = {
        "model": "nvidia/MiniMax-M3-NVFP4",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"thinking_mode": "enabled"},
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    # Dynamo's health entry can become visible just before the frontend has
    # finished adding the model route.  The first request then receives a
    # transient 404 even though the worker is healthy.  Retry only discovery
    # statuses; all payload/tool-call failures still fail immediately.
    for attempt in range(60):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 200:
                    raise RuntimeError(f"target returned HTTP {response.status}")
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in {404, 503} or attempt == 59:
                raise
            time.sleep(2)
    raise RuntimeError("target model route did not become ready")


# Dynamo-vLLM 1.3.1 does not reliably preserve the named-function form of
# tool_choice for MiniMax-M3.  The generic OpenAI `required` form produced
# structured calls in the prior capability run, so require a tool call here
# and validate the selected function below.
first = complete([user_message], tools=[tool], tool_choice="required")
capture_path.write_text(json.dumps(first, indent=2) + "\n")
assistant = first["choices"][0]["message"]
tool_calls = assistant.get("tool_calls") or []
matching_calls = [
    call
    for call in tool_calls
    if call.get("function", {}).get("name") == "lookup_subscriber"
]
if not matching_calls:
    raise SystemExit("tau2 capability gate did not receive a structured lookup_subscriber call")
call = matching_calls[0]
arguments = json.loads(call["function"]["arguments"])
if arguments.get("account_id") != "test-123":
    raise SystemExit("tau2 capability gate received the wrong tool arguments")

assistant_message = {
    "role": "assistant",
    "content": assistant.get("content"),
    "tool_calls": tool_calls,
}
tool_result = {
    "role": "tool",
    "tool_call_id": call["id"],
    "content": json.dumps({"account_id": "test-123", "status": "active"}),
}
second = complete([user_message, assistant_message, tool_result], tools=[tool])
if not second.get("choices"):
    raise SystemExit("tau2 capability gate tool-result continuation returned no choice")
print("tau2 structured tool-call and tool-result continuation gate passed")
PY
fi

# SciCode executes generated Python locally. Start the exact sandbox module,
# require its health endpoint, and score a deterministic code-execution smoke
# before allowing the benchmark harness to launch its own sandbox instance.
if [ "$task_name" = "ns_scicode" ]; then
    sandbox_log="${output_dir}/sandbox-capability.log"
    sandbox_result="${output_dir}/sandbox-capability.json"
    python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server >"$sandbox_log" 2>&1 &
    sandbox_pid=$!
    cleanup_sandbox() {
        kill "$sandbox_pid" 2>/dev/null || true
        wait "$sandbox_pid" 2>/dev/null || true
    }
    trap cleanup_sandbox EXIT

    sandbox_ready=false
    for _ in $(seq 1 30); do
        if curl --silent --fail --max-time 2 http://127.0.0.1:6000/health >/dev/null; then
            sandbox_ready=true
            break
        fi
        if ! kill -0 "$sandbox_pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if [ "$sandbox_ready" != "true" ]; then
        echo "SciCode local sandbox failed its startup gate" >&2
        exit 1
    fi

    curl --silent --show-error --fail \
        --output "$sandbox_result" \
        --max-time 60 \
        --header 'Content-Type: application/json' \
        --data '{"generated_code":"print(6 * 7)","timeout":30,"language":"python"}' \
        http://127.0.0.1:6000/execute
    python3 - "$sandbox_result" <<'PY'
import json
import pathlib
import sys

result = json.loads(pathlib.Path(sys.argv[1]).read_text())
if result.get("process_status") != "completed":
    raise SystemExit("SciCode sandbox smoke did not complete")
if result.get("stdout", "").strip() != "42" or result.get("stderr", ""):
    raise SystemExit("SciCode sandbox smoke produced an incorrect result")
print("SciCode local sandbox scored code-execution gate passed")
PY
    cleanup_sandbox
    trap - EXIT
fi

evaluator=$(command -v nemo-evaluator || command -v eval-factory)
"$evaluator" run_eval --run_config "$resolved_config"
