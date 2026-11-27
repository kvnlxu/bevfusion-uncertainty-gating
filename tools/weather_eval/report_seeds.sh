#!/usr/bin/env bash
# Summarise the multi-seed run (queue_seeds.sh) into work_dirs/gate/seeds_report.txt
# and print it. Safe to run while the queue is still going (shows progress).
#   bash tools/weather_eval/report_seeds.sh
cd "$(dirname "$0")/../.."
OUT=work_dirs/gate/seeds_report.txt
{
  echo "=== QUEUE PROGRESS ==="
  grep -E "^=== |FAILED|missing" work_dirs/gate/seeds_queue.out 2>/dev/null | tail -8

  echo; echo "=== POOLED TESTS: supervised spatial gate vs control, all seeds ==="
  for F in heavyfog clean; do
    python - "$F" <<'EOF'
import csv, sys
from pathlib import Path
name = sys.argv[1]
p = Path(f"work_dirs/weather_sweep/bootstrap_seeds_{name}.csv")
if not p.exists():
    print(f"{name}: not run yet")
    sys.exit()
rows = list(csv.DictReader(open(p)))
for r in rows:
    label = "POOLED" if r["run_a"].startswith("POOLED") else Path(r["run_a"]).name
    print(f"{name:>8} {label:<42} diff {100*float(r['diff']):+.2f} pts  "
          f"CI [{100*float(r['ci_low']):+.2f}, {100*float(r['ci_high']):+.2f}]  "
          f"p={float(r['p_value']):.3f}")
EOF
  done

  echo; echo "=== PER RUN: camera weight and mAP_present ==="
  python - <<'EOF'
import csv
from pathlib import Path
p = Path("work_dirs/weather_sweep/results.csv")
for r in csv.DictReader(open(p)):
    if (any(t in r["model"] for t in ("sup10", "ctrl"))
            and r["weather"] in ("clean", "fog")
            and float(r["camera_scale"] or 1) == 1):
        print(f"{r['model']:>26} {r['weather']:>5} s={float(r['severity']):.1f}  "
              f"w_cam={(r['gate_w_cam'] or '-')[:5]:>5}  mAP_present={float(r['mAP_present']):.3f}")
EOF
} > "$OUT" 2>&1
cat "$OUT"
