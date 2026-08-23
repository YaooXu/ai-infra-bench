#!/bin/sh
set -u

LOG_DIR=${BENCH_LOG_DIR:-/logs/verifier}
TESTS_DIR=${BENCH_TESTS_DIR:-/tests}
PYTHON=${BENCH_PYTHON:-python3}

case "$LOG_DIR" in
    /*) ;;
    *) echo "BENCH_LOG_DIR must be absolute" >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR"
# Never reuse output from an earlier Harbor trial.
rm -f \
    "$LOG_DIR/reward.json" \
    "$LOG_DIR/scoring.json" \
    "$LOG_DIR/reward.txt" \
    "$LOG_DIR/result.json" \
    "$LOG_DIR/verifier_result.json" \
    "$LOG_DIR/semantic_evidence.json" \
    "$LOG_DIR/semantic_hash.json" \
    "$LOG_DIR/diagnostic_evidence.json" \
    "$LOG_DIR/requirement_scores.json" \
    "$LOG_DIR/integrity_failure.json" \
    "$LOG_DIR/infra_error.json" \
    "$LOG_DIR/global_failure.json" \
    "$LOG_DIR/manifest_failure.json" \
    "$LOG_DIR/collection_errors.json" \
    "$LOG_DIR/scoring_failure.json" \
    "$LOG_DIR"/junit-*.xml

run_rc=0
BENCH_TESTS_DIR="$TESTS_DIR" "$PYTHON" -B -I -c '
import os
import runpy
import sys

tests_dir = os.environ["BENCH_TESTS_DIR"]
sys.path.insert(0, tests_dir)
runpy.run_path(tests_dir + "/run_verifier.py", run_name="__main__")
' || run_rc=$?

# A verifier crash must still create a fresh explicit failure record.
if [ ! -s "$LOG_DIR/reward.json" ]; then
    BENCH_LOG_DIR="$LOG_DIR" "$PYTHON" -B -I -c '
import json, os, pathlib
root = pathlib.Path(os.environ["BENCH_LOG_DIR"])
reward = {"reward": 0.0}
scoring = {
    "reward": 0.0,
    "raw_correctness": 0.0,
    "validity_gate": 0,
    "infra_error": 1,
    "release_eligible": 0,
    "failure_reason": "verifier_exited_without_reward",
}
(root / "reward.json").write_text(json.dumps(reward, sort_keys=True) + "\n")
(root / "scoring.json").write_text(json.dumps(scoring, sort_keys=True) + "\n")
'
fi

# Strictly validate the fresh result and emit Harbor JSON plus reward.txt.
if ! BENCH_LOG_DIR="$LOG_DIR" VERIFIER_RUN_RC="$run_rc" "$PYTHON" -B -I -c '
import json, math, os, pathlib
root = pathlib.Path(os.environ["BENCH_LOG_DIR"])
payload = json.loads((root / "reward.json").read_text())
reward = payload.get("reward")
if isinstance(reward, bool) or not isinstance(reward, (int, float)):
    raise ValueError("reward must be numeric")
reward = float(reward)
if not math.isfinite(reward) or reward not in (0.0, 1.0):
    raise ValueError("formal correctness reward must be binary")
if set(payload) != {"reward"}:
    raise ValueError("reward.json must contain only the formal reward")
scoring = json.loads((root / "scoring.json").read_text())
for name in ("validity_gate", "infra_error", "release_eligible"):
    if scoring.get(name) not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
raw = scoring.get("raw_correctness")
if isinstance(raw, bool) or not isinstance(raw, (int, float)):
    raise ValueError("raw_correctness must be numeric")
if not math.isfinite(float(raw)) or not 0.0 <= float(raw) <= 1.0:
    raise ValueError("raw_correctness must be finite and within [0, 1]")
if float(scoring.get("reward", -1)) != reward:
    raise ValueError("formal and diagnostic rewards disagree")
(root / "reward.txt").write_text(f"{reward:.6f}\n")
result = {
    "schema_version": "1.0",
    "verifier_exit_code": int(os.environ["VERIFIER_RUN_RC"]),
    "reward": payload,
    "scoring": scoring,
}
(root / "result.json").write_text(
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
)
'; then
    rm -f "$LOG_DIR/reward.txt" "$LOG_DIR/result.json"
    printf '%s\n' '0.000000' >"$LOG_DIR/reward.txt"
    printf '%s\n' '{"schema_version":"1.0","verifier_exit_code":2,"reward":{"reward":0.0},"scoring":{"reward":0.0,"raw_correctness":0.0,"validity_gate":0,"infra_error":1,"release_eligible":0,"failure_reason":"invalid_reward_format"}}' >"$LOG_DIR/result.json"
    exit 2
fi

exit 0
