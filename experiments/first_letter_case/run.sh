#!/usr/bin/env bash
# Verbatim payloads vs label-independent first-letter case, same model and data.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DPID="$HERE/../../dPID"
export OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
for seed in ${SEEDS:-0 1 2}; do
  for arm in verbatim redrawn; do
    (cd "$DPID" && python tests/first_letter_shortcut.py --arm "$arm" --seed "$seed" --epochs 8 \
        --out "$HERE/${arm}_seed$seed.json" > "$HERE/${arm}_seed$seed.log" 2>&1) &
  done
  wait
done
