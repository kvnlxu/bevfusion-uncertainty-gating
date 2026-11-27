#!/usr/bin/env bash
# Multi-seed check of the Phase 4 result: does the supervised spatial gate beat
# the fine-tuning control (gate locked at w = 1) across training seeds?
#
# For each seed: trains the supervised spatial gate (sup10) and the control,
# evaluates both on clean data and heavy fog (1.5-3.0), then pools the
# spatial-vs-control comparison over seed 0 (already trained) and these seeds.
# Finished trainings and evaluations are skipped, so it can be restarted.
#
# Run from the repository root (about 1 h per seed):
#   nohup bash tools/weather_eval/queue_seeds.sh > work_dirs/gate/seeds_queue.out 2>&1 &
#   nohup bash tools/weather_eval/queue_seeds.sh 3 4 > work_dirs/gate/seeds_queue.out 2>&1 &   # other seeds
# Then: bash tools/weather_eval/report_seeds.sh
set -u
cd "$(dirname "$0")/../.."

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2)
S=work_dirs/weather_sweep

for SEED in "${SEEDS[@]}"; do
  for M in sup10 ctrl; do
    D=work_dirs/gate/${M}_seed${SEED}
    if [ "$M" = sup10 ]; then
      G=spatial
      GEN_ARGS=(--gate spatial --reliability-weight 1.0 --gate-lr-mult 10)
    else
      G=modality
      GEN_ARGS=(--freeze-gate)
    fi
    echo "=== $(date '+%H:%M:%S') seed $SEED: $M ==="
    mkdir -p "$D"
    if grep -qs "Saving checkpoint at 6 epochs" "$D/train.out"; then
      echo "already trained: $D"
    else
      python tools/weather_eval/make_gate_config.py --output-dir "$D" "${GEN_ARGS[@]}" --seed "$SEED" || continue
      python tools/weather_eval/train_gate.py "$D/config.py" > "$D/train.out" 2>&1
      if ! grep -q "Saving checkpoint at 6 epochs" "$D/train.out"; then
        echo "TRAINING FAILED: $D (see $D/train.out)"
        continue
      fi
    fi
    python tools/weather_eval/sweep.py --gate "$G" --checkpoint "$D/latest.pth" \
      --run-tag "${M}s${SEED}" --weathers fog --severities 1.5 2.0 2.5 3.0
  done
done

# Pool spatial vs control over seed 0 and every seed whose runs exist.
FOG=(); CLEAN=()
for SEED in 0 "${SEEDS[@]}"; do
  if [ "$SEED" = 0 ]; then A=sup10_gated_spatial_; B=ctrl_gated_modality_
  else A=sup10s${SEED}_gated_spatial_; B=ctrls${SEED}_gated_modality_; fi
  if [ ! -f "$S/${A}fog_300_both_s42/metrics.json" ] || [ ! -f "$S/${B}fog_300_both_s42/metrics.json" ]; then
    echo "seed $SEED: results missing, left out of the pooled tests"
    continue
  fi
  for F in fog_150 fog_200 fog_250 fog_300; do
    FOG+=(--pair "$S/${A}${F}_both_s42" "$S/${B}${F}_both_s42")
  done
  CLEAN+=(--pair "$S/${A}clean_000_both_s42" "$S/${B}clean_000_both_s42")
done
echo "=== $(date '+%H:%M:%S') pooled tests ==="
python tools/weather_eval/bootstrap.py --pool "${FOG[@]}" --out "$S/bootstrap_seeds_heavyfog.csv"
python tools/weather_eval/bootstrap.py --pool "${CLEAN[@]}" --out "$S/bootstrap_seeds_clean.csv"
echo "=== $(date '+%H:%M:%S') done ==="
