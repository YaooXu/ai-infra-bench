#!/usr/bin/env bash
###############################################################################
# 参考解法 (reference solution)
# 状态: 未开发 (curator 后续补充)
#
# 该题目当前以 correctness 评测: verifier 用默认编译模式启动 agent 修改后的
# 服务, 比对 grounding bbox 与参考。参考解法用于验证:
#   - base 状态必失败
#   - 参考解法必通过
#   - no-op / 硬编码 / 常见错误解必失败
#
# 参考修复方向(供 curator 参考, 不泄露给 agent):
#   issue #29595 根因在 torch 2.9.0 + triton 3.5.0 的 Inductor 编译数值错误
#   (pytorch/pytorch#167339), 上游无 vLLM 修复 PR, 需在 vLLM 源码层设计修复,
#   例如针对编译路径中数值错误的算子做规避/改写, 且不全局关闭编译。
###############################################################################
echo "TODO: reference solution not yet implemented"
exit 1
