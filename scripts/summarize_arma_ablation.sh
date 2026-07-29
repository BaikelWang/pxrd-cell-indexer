#!/usr/bin/env bash
# Print the Arm A 100k ablation comparison table (safe to run any time, mid-run included).
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json
import re
from pathlib import Path

# (label, result dir, training log)
ROWS = [
    ("自研 peaktf",        "results/flow_seedgen/pilot100k_equiv_off",           "results/flow_seedgen/logs/pilot_equiv_off.log"),
    ("ArmA 全冻 (对照)",   "results/flow_seedgen/armA_pilot100k_equiv_off",      "logs/armA_pilot100k_equiv_off.log"),
    ("A 双解冻",           "results/flow_seedgen/armA_pilot100k_unfreeze_both",  "logs/armA_pilot100k_unfreeze_both.log"),
    ("C 解冻 CSPNet",      "results/flow_seedgen/armA_pilot100k_unfreeze_decoder", "logs/armA_pilot100k_unfreeze_decoder.log"),
    ("D 砍掉 CSPNet",      "results/flow_seedgen/armA_pilot100k_no_cspnet",      "logs/armA_pilot100k_no_cspnet.log"),
    ("F D+大encoder(scratch)", "results/flow_seedgen/armA_pilot100k_scratch_d256L6", "logs/armA_pilot100k_scratch_d256L6.log"),
    ("E Fourier 位置编码", "results/flow_seedgen/armA_pilot100k_fourier_pos",    "logs/armA_pilot100k_fourier_pos.log"),
]

EVAL = re.compile(r"ep(\d+).*?cov@1=(\d+)%\s+@20=(\d+)%\s+@100=(\d+)%")
EPOCH = re.compile(r"^ep(\d+)\s+train=", re.M)


def from_log(log):
    """Return (best_tuple, last_epoch) parsed from a training log."""
    p = Path(log)
    if not p.exists():
        return None, None
    text = p.read_text(errors="ignore")
    eps = [int(m) for m in EPOCH.findall(text)]
    last = max(eps) if eps else None
    best = None
    for ep, c1, c20, c100 in EVAL.findall(text):
        cur = (int(c100), int(ep), int(c1), int(c20))
        if best is None or cur[0] > best[0]:
            best = cur
    return best, last


print("\n===== Arm A 100k 消融: MP100 primitive library@100 =====")
print(f"{'run':20s} {'best@100':>9s} {'ep':>4s} {'@1':>5s} {'@20':>5s}  状态")
for name, d, log in ROWS:
    meta = Path(d) / "best_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        c = m.get("coverage", {})
        b100 = f"{100*m['library_strict']:.0f}%"
        ep = str(m.get("epoch", "?"))
        c1 = f"{100*c.get('1', 0):.0f}%"
        c20 = f"{100*c.get('20', 0):.0f}%"
        state = "完成"
    else:
        best, last = from_log(log)
        if best is None:
            b100 = ep = c1 = c20 = "—"
            state = "运行中(未出eval)" if last else "未开始"
        else:
            b100, ep, c1, c20 = f"{best[0]}%", str(best[1]), f"{best[2]}%", f"{best[3]}%"
            state = f"运行中 ep{last}/60"
    print(f"{name:20s} {b100:>9s} {ep:>4s} {c1:>5s} {c20:>5s}  {state}")

print("\nB(只解冻 Encoder) 已移出队列: Bert encoder 仅 36,928 参数,")
print("且位于 2θ→.long() 的 1° 量化下游, 无法突破输入信息天花板。")
print("F 为 scratch 对照(encoder 4.98M 从零训, 预训练权重无法加载), 非迁移学习。")
print("\n2x2 析因:        int位置(1°bin)    Fourier连续位置")
print("  encoder 37k         D                 E")
print("  encoder ~5M         F            ≈ 自研 peaktf 51%")
PY
