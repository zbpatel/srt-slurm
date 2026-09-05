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
        mmlu_pro_aa_v3)
            recovery_source="$RECOVERY_ACCURACY_DIR/mmlu_pro_aa_v3/cache"
            recovery_target="$output_dir/mmlu_pro_aa_v3/cache"
            ;;
        ns_aa_lcr)
            recovery_source="$RECOVERY_ACCURACY_DIR/tmp-eval-results/aalcr"
            recovery_target="$output_dir/tmp-eval-results/aalcr"
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

    if [ "$task_name" = "mmlu_pro_aa_v3" ]; then
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
    else
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

# AA-LCR uses an external judge.  Read the credential from the ephemeral
# GitLab Secure File staged by AIB; never place it in the recipe, command line,
# stdout, or result artifacts.  A real chat request catches authorization and
# model-name errors before the expensive generation phase begins.
if [ "$task_name" = "ns_aa_lcr" ]; then
    : "${JUDGE_API_KEY_FILE:?AA-LCR requires JUDGE_API_KEY_FILE}"
    if [ ! -r "$JUDGE_API_KEY_FILE" ]; then
        echo "Judge key file is not readable" >&2
        exit 2
    fi
    INFERENCE_API_KEY=$(<"$JUDGE_API_KEY_FILE")
    export INFERENCE_API_KEY
    if [ -z "$INFERENCE_API_KEY" ]; then
        echo "Judge key file is empty" >&2
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
        echo "Qwen-235B judge authorization probe failed with HTTP ${judge_status}" >&2
        exit 1
    fi
    echo "Qwen-235B judge authorization probe passed"
fi

evaluator=$(command -v nemo-evaluator || command -v eval-factory)
"$evaluator" run_eval --run_config "$resolved_config"
