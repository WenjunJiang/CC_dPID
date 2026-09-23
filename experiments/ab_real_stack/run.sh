#!/usr/bin/env bash
# Old vs new segment objective on the production stack (tiny model, CPU).
# OLD_CODE must hold the dPID tree before the fix: git archive d843fb1 dPID | tar -x -C "$OLD_CODE/.."
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DPID="$HERE/../../dPID"
OLD_CODE=${OLD_CODE:?set OLD_CODE to the pre-fix dPID directory}
export OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
for seed in ${SEEDS:-0 1 2}; do
  (cd "$DPID" && DPID_CODE_DIR="$OLD_CODE" python tests/ab_segment_objective.py \
      --arm old --seed "$seed" --out "$HERE/old_seed$seed.json" > "$HERE/old_seed$seed.log" 2>&1) &
  (cd "$DPID" && python tests/ab_segment_objective.py \
      --arm new --seed "$seed" --out "$HERE/new_seed$seed.json" > "$HERE/new_seed$seed.log" 2>&1) &
  wait
done
