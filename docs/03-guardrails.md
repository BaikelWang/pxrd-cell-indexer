# Step 4 — 自动化护栏

> **状态**：🟢 脚手架已就绪  
> **最后更新**：2026-07-06

---

## 一键验证

```bash
cd Task/PRXD-Cell-indexing-model-0706

# 安装开发依赖
pip install -e ".[dev]"

# 运行全部检查（lint + typecheck + test）
make test
```

## 可用 Make 目标

| 命令 | 说明 |
|---|---|
| `make test` | ruff check + mypy + pytest（默认入口） |
| `make lint` | 仅 ruff 静态检查 |
| `make typecheck` | 仅 mypy 类型检查 |
| `make pytest` | 仅运行 pytest |

## 工具链

| 工具 | 配置 | 用途 |
|---|---|---|
| **ruff** | `pyproject.toml [tool.ruff]` | lint + format |
| **mypy** | `pyproject.toml [tool.mypy]` | 静态类型检查 |
| **pytest** | `pyproject.toml [tool.pytest]` | 测试框架 |

## 冒烟测试

`tests/test_smoke.py` 验证：

1. 包可 import
2. 核心类型可实例化
3. pipeline stub 可调用且不抛异常

## 待 Step 4 完善

- [ ] 锁定 `requirements.txt` 精确版本
- [ ] 添加 pre-commit 配置
- [ ] 添加 CI workflow（如有远程仓库需求）
- [ ] 添加最小端到端集成测试（依赖真实数据路径确认后）
