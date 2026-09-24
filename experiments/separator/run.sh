#!/usr/bin/env bash
# Fixed paragraph separator vs a label-independent separator draw, same model and data.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DPID="$HERE/../../dPID"
export OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
for seed in ${SEEDS:-0 1 2}; do
  for arm in paragraph mixed; do
    (cd "$DPID" && python tests/separator_shift.py --arm "$arm" --seed "$seed" --epochs 8 \
        --out "$HERE/${arm}_seed$seed.json" > "$HERE/${arm}_seed$seed.log" 2>&1) &
  done
  wait
done
