# 脚本入口

Step 5 起在此添加 CLI 脚本，例如：

- `investigate_10k_sample.py` — M1.2 只读数据调研：20 万随机池统计 + 10k 分层抽样（已完成）
- `train.py` — 训练入口
- `eval.py` — 批量评测（可参考 `CNRS/code/1_eval.py`）
- `infer.py` — 单样本推理

当前除调研脚本外仍为空，等待 Step 3 骨架确认后再添加训练/评测入口。
