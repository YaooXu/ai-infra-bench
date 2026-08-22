#!/usr/bin/env bash
###############################################################################
# Harbor verifier: Qwen3-VL grounding 坐标偏差修复评测
#
# 流程:
#   1. 用 agent 修改后的 vLLM 源码 (/workspace/vllm-src) 以默认编译模式启动服务
#      (不传 --enforce-eager —— 这正是被修复的对象)
#   2. 等待 /health 就绪
#   3. 逐个执行测试用例: 提交 grounding 请求 -> 解析 bbox -> 与期望对比
#      (取面积最大检测框, 与预存参考的任一坐标差 > 阈值即 FAIL)
#
# 测试用例目录结构 (tests/required/ 与 tests/heldout/ 同构):
#   <case-name>/
#     test_image.png     # 待检测图片
#     query_bbox.py      # 请求脚本 (OpenAI 兼容 API)
#     expected_bbox.json # {"prompt": "...", "expected_bbox": [x1,y1,x2,y2], "threshold_px": N}
#
# 用法: test.sh <case_dir...>
#   例如: test.sh tests/required/case_wikipedia
#   Verifier 传入 required 与 heldout 的全部用例目录。
###############################################################################
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/Qwen3-VL-30B-A3B-Thinking}"
SERVED_NAME="qwen-vl"
PORT="${PORT:-8000}"
API_BASE="http://localhost:${PORT}/v1"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVER_WAIT_SEC=600
KEEP_SERVER="${KEEP_SERVER:-0}"   # 调试用: 1 = 结束后保留服务

log() { echo -e "\033[1;36m[$(date '+%H:%M:%S')]\033[0m $*"; }
die() { echo -e "\033[1;31m[错误]\033[0m $*" >&2; exit 1; }

[ -d "$MODEL_DIR" ] || die "找不到模型目录: $MODEL_DIR"
[ -d "/workspace/vllm-src" ] || die "找不到 vLLM 源码: /workspace/vllm-src"

# ------------------------- 服务管理 -------------------------
SERVER_PID=""
start_server() {
  log "启动默认编译模式服务 (模型 $MODEL_DIR, 端口 $PORT) ..."
  CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=fork \
    nohup python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_DIR" \
      --served-model-name "$SERVED_NAME" \
      --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --reasoning_parser deepseek_r1 --no-enable-prefix-caching \
      --port "$PORT" --max-model-len "$MAX_MODEL_LEN" \
      > /tmp/vllm_verify_server.log 2>&1 &
  SERVER_PID=$!
}

wait_health() {
  local t=0
  log "等待服务就绪 (最多 ${SERVER_WAIT_SEC}s) ..."
  while [ "$t" -lt "$SERVER_WAIT_SEC" ]; do
    if curl -sf --max-time 3 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      log "服务已就绪"
      return 0
    fi
    # 服务进程退出 -> 立即失败
    kill -0 "$SERVER_PID" 2>/dev/null || {
      die "服务进程已退出, 日志尾部:\n$(tail -20 /tmp/vllm_verify_server.log)"
    }
    sleep 5; t=$((t+5))
  done
  die "服务在 ${SERVER_WAIT_SEC}s 内未就绪, 日志尾部:\n$(tail -20 /tmp/vllm_verify_server.log)"
}

cleanup() {
  [ "$KEEP_SERVER" = "1" ] && return 0
  log "清理服务 (pid $SERVER_PID) ..."
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -f "vllm.entrypoints.openai.api_server --model $MODEL_DIR" 2>/dev/null || true
  sleep 2
}
trap cleanup EXIT

# ------------------------- 单个用例 -------------------------
run_case() {
  local case_dir="$1"
  local img="$case_dir/test_image.png"
  local script="$case_dir/query_bbox.py"
  local meta="$case_dir/expected_bbox.json"

  [ -f "$img" ]    || die "用例缺少图片: $img"
  [ -f "$script" ] || die "用例缺少 query_bbox.py: $script"
  [ -f "$meta" ]   || die "用例缺少 expected_bbox.json: $meta"

  local prompt expected threshold
  prompt=$(python3 -c "import json;print(json.load(open('$meta'))['prompt'])")
  expected=$(python3 -c "import json;print(json.load(open('$meta'))['expected_bbox'])")
  threshold=$(python3 -c "import json;print(json.load(open('$meta'))['threshold_px'])")

  log "【用例 $(basename "$case_dir")】提交 grounding 请求: '$prompt'"
  local details="/tmp/verify_details.json"
  python3 "$script" "$img" "$prompt" \
    --api-base "$API_BASE" --model "$SERVED_NAME" \
    --seed 1 --max-tokens 1024 \
    --save-generation-details "$details" > /tmp/verify_out.txt 2>&1 \
    || die "推理失败, 见 /tmp/verify_out.txt"

  python3 - "$details" "$expected" "$threshold" <<'EOF'
import json, sys

d = json.load(open(sys.argv[1]))
expected = json.loads(sys.argv[2])
threshold = float(sys.argv[3])

# 取面积最大的检测框 (与复现脚本一致)
best, best_area = None, -1
for det in d.get("detections", []):
    b = det.get("bbox_2d")
    if not b or len(b) != 4:
        continue
    area = (b[2] - b[0]) * (b[3] - b[1])
    if area > best_area:
        best_area, best = area, b

if best is None:
    print(f"FAIL: 未检出 bbox")
    sys.exit(1)

maxdiff = max(abs(a - b) for a, b in zip(best, expected))
print(f"  检测 bbox: {best}")
print(f"  期望 bbox: {expected}")
print(f"  最大坐标差: {maxdiff}px (阈值 {threshold:.0f}px)")

if maxdiff <= threshold:
    print("  PASS")
    sys.exit(0)
else:
    print(f"  FAIL: 坐标偏差超出阈值 (bug 未修复)")
    sys.exit(1)
EOF
}

# ------------------------- 主流程 -------------------------
[ $# -gt 0 ] || die "用法: test.sh <case_dir...>"
case_dirs=("$@")

start_server
wait_health

pass_cnt=0; fail_cnt=0
for case_dir in "${case_dirs[@]}"; do
  if run_case "$case_dir"; then
    pass_cnt=$((pass_cnt+1))
  else
    fail_cnt=$((fail_cnt+1))
  fi
done

echo
echo "======================================================================"
echo "  结果: $pass_cnt 通过 / $fail_cnt 失败 (共 $((pass_cnt+fail_cnt)) 用例)"
echo "======================================================================"

[ "$fail_cnt" -eq 0 ]
