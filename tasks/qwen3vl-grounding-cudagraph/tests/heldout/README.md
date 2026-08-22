# heldout/ —— 留出测试（发布后由 curator 补充, 不暴露给 agent）

留出测试与 required/ 同构（每个用例目录含 `test_image.png` + `query_bbox.py` +
`expected_bbox.json`），但使用**新的测试图片与 prompt**，防止 agent 硬编码到
公开示例。

当前状态：**待补充**。至少需要 1-2 个额外的 grounding 用例，例如：

1. 另一张 Wikipedia 风格截图（不同版式/元素），prompt 类似 "Locate the X button"
2. 一张不同来源的 UI 截图（如软件界面/网页表单）

每个用例的 `expected_bbox.json` 需由 curator 在 **eager 模式**下生成
（用 `--enforce-eager` 启动 vLLM 服务跑出正确结果后提取），并单独保存参考
（reference）供评测比对。

生成流程（参考复现仓库 `reproduce_grounding_bug.sh`）:
1. 用 eager 模式启动服务, 对 heldout 图片提交 grounding 请求
2. 取面积最大检测框作为 `expected_bbox` (与 test.sh 的判定逻辑一致)
3. 确认该 bbox 与 default 模式结果偏差 > 3px (即该用例确实触发 bug)
4. 将结果写入本目录用例的 `expected_bbox.json`

注意: heldout 用例数据在题目发布前不得进入 agent 可见的任何文件。
