# MiniMax-M3 NeMo Evaluator jobs

These templates are the `config_ef.yaml` payloads produced by a dry run of
NeMo Evaluator Launcher `0.2.7+20260902.673edc82` with the internal task pack
`0.3.178+20260805`.  `run_nel_eval.sh` replaces only the live srt-slurm
frontend URL and the run-local output directory.

All tasks use MiniMax-M3 thinking mode, strip and track
`<mm:think>...</mm:think>`, and generate with temperature `1.0`, top-p `0.95`,
and a 65,536-token cap. MMLU-Pro uses the AA-v3 task with one sample.
GPQA-Diamond uses the AA-v3 task with 16 samples and 128-way dispatch. AA-LCR
uses 16 samples with 64-way generation and judge dispatch, matching the
observed headroom of the TP4 server, and the authorized
`nvidia/qwen/eccn-qwen-235b` judge.  The judge credential must be exposed as a
read-only `JUDGE_API_KEY_FILE`; the runner reads it without copying the secret
into results. Telecom uses the same authorized Qwen-235B endpoint as its user
simulator, runs the full base set with three trials and 64-way simulation, and
requires a structured tool-call round trip before evaluation. SciCode uses its
canonical local sandbox at pass@1 with 16-way generation and requires a scored
code-execution smoke before evaluation.
