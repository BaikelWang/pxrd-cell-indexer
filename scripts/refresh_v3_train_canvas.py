#!/usr/bin/env python3
"""Regenerate the v3 training-curves canvas from metrics.csv / history.json.

Canvas files cannot poll the filesystem themselves, so this script rewrites
``pxrd-indexer-full6m-v3-curves.canvas.tsx`` whenever metrics grow:

  while true; do python3 scripts/refresh_v3_train_canvas.py; sleep 30; done
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / os.environ.get(
    "RUN", "results/flow_seedgen/pxrd_indexer_full6m_v5_imin"
)
_CANVAS_NAME = os.environ.get(
    "CANVAS_NAME", f"{RUN.name.replace('_', '-')}-curves.canvas.tsx"
)
CANVAS = Path("/root/.cursor/projects/nanolab-users-wyx/canvases") / _CANVAS_NAME
EPOCHS_TOTAL = int(os.environ.get("EPOCHS_TOTAL", "60"))

TEMPLATE = r'''import {
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  LineChart,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

/**
 * Auto-refreshed from results/flow_seedgen/<run>/metrics.csv
 * by scripts/refresh_v3_train_canvas.py — do not hand-edit ROWS.
 */

type Row = {
  epoch: number;
  train_loss: number | null;
  valid_loss: number | null;
  hit02: number | null;
  hit1: number | null;
  macro: number | null;
  worst: number | null;
  select: number | null;
  mp100: number | null;
  elapsed_min: number;
};

const RUN = "__RUN_NAME__";
const TITLE = "__TITLE__";
const UPDATED = "__UPDATED__";
const STATUS: "waiting" | "running" | "done" = "__STATUS__";
const WORLD = __WORLD__;
const BS_PER_GPU = __BS__;
const GLOBAL_BATCH = __GBS__;
const LR = __LR__;
const AMP = "__AMP__";
const EPOCHS_TOTAL = __EPOCHS_TOTAL__;
const BEST_EP: number | null = __BEST_EP__;
const BEST_SELECT: number | null = __BEST_SELECT__;

const ROWS: Row[] = [
__ROWS__
];

const MP100_POINTS: { epoch: number; value: number }[] = [
__MP100__
];

const pct = (v: number | null | undefined) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(1)}%`;
const num = (v: number | null | undefined, d = 4) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(d);

export default function PxrdIndexerFull6mV3Curves() {
  const theme = useHostTheme();
  const muted = theme.text.secondary;
  const latest = ROWS.length ? ROWS[ROWS.length - 1] : null;
  const cats = ROWS.map((r) => String(r.epoch));
  const mpCats = MP100_POINTS.map((p) => String(p.epoch));

  const statusTone =
    STATUS === "done" ? "success" : STATUS === "running" ? "info" : "warning";

  let callout = "Training started; waiting for ep001 metrics.";
  if (STATUS === "done" && latest) {
    callout = `Finished ep${latest.epoch}/${EPOCHS_TOTAL}. Best valid_macro@0.2% = ${pct(BEST_SELECT)} @ ep${BEST_EP}.`;
  } else if (STATUS === "running" && latest) {
    callout = `Running ep${latest.epoch}/${EPOCHS_TOTAL}. Best so far valid_macro@0.2% = ${pct(BEST_SELECT)} @ ep${BEST_EP}. Latest select=${pct(latest.select)} macro=${pct(latest.macro)} worst=${pct(latest.worst)}.`;
  }

  const lastMp =
    latest?.mp100 != null
      ? pct(latest.mp100)
      : MP100_POINTS.length
        ? pct(MP100_POINTS[MP100_POINTS.length - 1].value)
        : null;

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1040 }}>
      <Stack gap={6}>
        <H1>{TITLE}</H1>
        <Text style={{ color: muted }}>
          Run: {RUN} · {WORLD}-GPU DDP · per-GPU bs={BS_PER_GPU} (global {GLOBAL_BATCH}) ·
          lr={LR} · amp={AMP} · select=valid_macro@0.2% n=700 · MP100 every 3ep (report only)
        </Text>
        <Text size="small" style={{ color: muted }}>
          Auto-refresh snapshot: {UPDATED} · status={STATUS} · epochs={ROWS.length}/{EPOCHS_TOTAL}
        </Text>
      </Stack>

      <Callout tone={statusTone}>{callout}</Callout>

      <Grid columns={4} gap={12}>
        <Stat label="Status" value={STATUS} tone={statusTone} />
        <Stat
          label={BEST_EP != null ? `Best macro@0.2% (ep${BEST_EP})` : "Best macro@0.2%"}
          value={pct(BEST_SELECT)}
          tone="success"
        />
        <Stat
          label="Latest worst-system"
          value={pct(latest?.worst)}
          tone={latest && latest.worst != null && latest.worst < 0.2 ? "warning" : undefined}
        />
        <Stat
          label="Wall / last MP100"
          value={
            latest
              ? `${latest.elapsed_min.toFixed(0)}m` + (lastMp ? ` · ${lastMp}` : "")
              : "—"
          }
        />
      </Grid>

      <Divider />

      {ROWS.length > 0 ? (
        <>
          <H2>Loss</H2>
          <LineChart
            categories={cats}
            series={[
              { name: "train_loss", data: ROWS.map((r) => r.train_loss ?? NaN) },
              { name: "valid_loss", data: ROWS.map((r) => r.valid_loss ?? NaN) },
            ]}
            height={220}
            beginAtZero={false}
          />
          <Text size="small" style={{ color: muted }}>
            Source: {RUN}/metrics.csv · x=epoch · y=loss
          </Text>

          <H2>Selection metrics (stratified valid n=700)</H2>
          <LineChart
            categories={cats}
            series={[
              {
                name: "valid_macro @0.2% (select)",
                data: ROWS.map((r) => (r.select ?? NaN) * 100),
                tone: "info",
              },
              {
                name: "worst system @0.2%",
                data: ROWS.map((r) => (r.worst ?? NaN) * 100),
                tone: "warning",
              },
              {
                name: "pooled <0.2%",
                data: ROWS.map((r) => (r.hit02 ?? NaN) * 100),
              },
              {
                name: "pooled <1%",
                data: ROWS.map((r) => (r.hit1 ?? NaN) * 100),
              },
            ]}
            height={240}
            valueSuffix="%"
            yMin={0}
            yMax={100}
          />
          <Text size="small" style={{ color: muted }}>
            select_score = valid_macro hit rate at 0.2% aligned length error. MP100 does not drive best.pt.
          </Text>
        </>
      ) : null}

      {MP100_POINTS.length > 0 ? (
        <>
          <H2>MP100 library_strict (every 3 epochs, report only)</H2>
          <LineChart
            categories={mpCats}
            series={[
              {
                name: "mp100 library_strict",
                data: MP100_POINTS.map((p) => p.value * 100),
                tone: "success",
              },
            ]}
            height={200}
            valueSuffix="%"
            yMin={0}
            yMax={100}
          />
        </>
      ) : null}

      {ROWS.length > 0 ? (
        <>
          <H2>Recent epochs</H2>
          <Table
            headers={[
              "ep",
              "train",
              "valid",
              "macro@0.2%",
              "worst@0.2%",
              "pooled<0.2%",
              "pooled<1%",
              "mp100",
              "elapsed",
            ]}
            rows={ROWS.slice(-12).map((r) => [
              String(r.epoch) + (r.epoch === BEST_EP ? " *" : ""),
              num(r.train_loss),
              num(r.valid_loss),
              pct(r.macro),
              pct(r.worst),
              pct(r.hit02),
              pct(r.hit1),
              pct(r.mp100),
              `${r.elapsed_min.toFixed(1)}m`,
            ])}
            columnAlign={[
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
            ]}
            striped
          />
        </>
      ) : null}
    </Stack>
  );
}
'''


def load_rows() -> list[dict]:
    metrics = RUN / "metrics.csv"
    hist = RUN / "history.json"
    if metrics.exists() and metrics.stat().st_size > 0:
        with metrics.open(newline="") as fh:
            return list(csv.DictReader(fh))
    if hist.exists():
        return json.loads(hist.read_text())
    return []


def fnum(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def lit(v: float | None) -> str:
    return "null" if v is None else f"{v:.6g}"


def main() -> None:
    rows = load_rows()
    args = json.loads((RUN / "args.json").read_text()) if (RUN / "args.json").exists() else {}
    # Defaults match plan B (v4 wide) when args.json is not yet written.
    _is_v4 = "v4" in RUN.name
    world = int(args.get("world_size", 4))
    bs = int(args.get("batch_size", 1024 if _is_v4 else 512))
    gbs = bs * world * int(args.get("grad_accum", 1))

    parsed = []
    for r in rows:
        parsed.append(
            {
                "epoch": int(float(r["epoch"])),
                "train_loss": fnum(r, "train_loss"),
                "valid_loss": fnum(r, "valid_loss"),
                "hit02": fnum(r, "valid_hit_0.2pct"),
                "hit1": fnum(r, "valid_hit_1pct"),
                "macro": fnum(r, "valid_macro_hit") or fnum(r, "select_score"),
                "worst": fnum(r, "valid_worst_system_hit"),
                "select": fnum(r, "select_score"),
                "mp100": fnum(r, "mp100_library_strict"),
                "elapsed_min": (fnum(r, "elapsed_s") or 0.0) / 60.0,
            }
        )

    row_lits = []
    for p in parsed:
        row_lits.append(
            "  { "
            + f"epoch: {p['epoch']}, "
            + f"train_loss: {lit(p['train_loss'])}, "
            + f"valid_loss: {lit(p['valid_loss'])}, "
            + f"hit02: {lit(p['hit02'])}, "
            + f"hit1: {lit(p['hit1'])}, "
            + f"macro: {lit(p['macro'])}, "
            + f"worst: {lit(p['worst'])}, "
            + f"select: {lit(p['select'])}, "
            + f"mp100: {lit(p['mp100'])}, "
            + f"elapsed_min: {p['elapsed_min']:.3f}"
            + " }"
        )

    mp_lits = [
        f"  {{ epoch: {p['epoch']}, value: {lit(p['mp100'])} }}"
        for p in parsed
        if p["mp100"] is not None
    ]

    latest_ep = parsed[-1]["epoch"] if parsed else 0
    status = "waiting" if not parsed else ("done" if latest_ep >= EPOCHS_TOTAL else "running")
    best = max((p for p in parsed if p["select"] is not None), key=lambda p: p["select"], default=None)

    run_name = RUN.name
    if "v4" in run_name and "lr2e3" in run_name:
        title = "PXRD-indexer full6m v4 wide (lr=2e-3) — training curves"
    elif "v4" in run_name:
        title = "PXRD-indexer full6m v4 wide — training curves"
    else:
        title = "PXRD-indexer full6m v3 — training curves"
    text = (
        TEMPLATE.replace("__RUN_NAME__", run_name)
        .replace("__TITLE__", title)
        .replace("__UPDATED__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
        .replace("__STATUS__", status)
        .replace("__WORLD__", str(world))
        .replace("__BS__", str(bs))
        .replace("__GBS__", str(gbs))
        .replace("__LR__", str(args.get("lr", 0.008 if _is_v4 else 0.004)))
        .replace("__AMP__", str(args.get("amp", "bf16")))
        .replace("__EPOCHS_TOTAL__", str(EPOCHS_TOTAL))
        .replace("__BEST_EP__", str(best["epoch"]) if best else "null")
        .replace("__BEST_SELECT__", lit(best["select"]) if best else "null")
        .replace("__ROWS__", (",\n".join(row_lits) + "\n") if row_lits else "")
        .replace("__MP100__", (",\n".join(mp_lits) + "\n") if mp_lits else "")
    )
    # Keep empty arrays on one line for cleaner TS.
    import re

    text = re.sub(r"const ROWS: Row\[\] = \[\s*\];", "const ROWS: Row[] = [];", text)
    text = re.sub(
        r"const MP100_POINTS: \{ epoch: number; value: number \}\[\] = \[\s*\];",
        "const MP100_POINTS: { epoch: number; value: number }[] = [];",
        text,
    )

    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    if CANVAS.exists() and CANVAS.read_text() == text:
        print(f"unchanged n={len(parsed)}")
        return
    CANVAS.write_text(text)
    print(f"wrote {CANVAS.name} n={len(parsed)} latest_ep={latest_ep} status={status}")


if __name__ == "__main__":
    main()
