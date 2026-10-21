#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CrossFlowIE
PKG=$ROOT/FT_NightVD_Support_v1
OUT=$ROOT/analysis-output/ft_nightvd_support_v1
VAL=$ROOT/data/dataset/uav/synthetic_smoke_val_v2
export PYTHONPATH="$ROOT/DehazeFormer-main:$PKG/source${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
(cd "$PKG" && sha256sum -c SHA256SUMS.txt)
printf '%s  %s\n' '943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128' "$VAL/generation_manifest.jsonl" | sha256sum -c -

for seed in 3407 2026 8888; do
  for arm in default expanded; do
    home=$OUT/FT_${arm}_seed${seed}
    test -f "$home/run_config.json"
    for epoch in 2 4 6 8 10; do test -f "$home/epoch_${epoch}.pth"; done
  done
done

python - "$OUT" <<'PY'
import json, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
for seed in (3407, 2026, 8888):
    steps = {}
    configs = {}
    for arm in ('default', 'expanded'):
        home = root / f'FT_{arm}_seed{seed}'
        config = json.loads((home / 'run_config.json').read_text())
        config['args'].pop('save_dir')
        config.pop('command')
        configs[arm] = config
        steps[arm] = [int(torch.load(home / f'epoch_{e}.pth', map_location='cpu')['global_step'])
                      for e in (2, 4, 6, 8, 10)]
    if configs['default'] != configs['expanded']:
        raise RuntimeError(f'Run-config mismatch for seed {seed}')
    if steps['default'] != steps['expanded']:
        raise RuntimeError(f'Optimizer-step mismatch for seed {seed}: {steps}')
    print('PAIRED_STEPS_PASS', seed, steps['default'])
PY

for seed in 3407 2026 8888; do
  for arm in default expanded; do
    home=$OUT/FT_${arm}_seed${seed}
    for epoch in 2 4 6 8 10; do
      dest=$home/epoch${epoch}
      test ! -e "$dest" || { echo "Refusing existing evaluation: $dest" >&2; exit 1; }
      python "$ROOT/eval_protocol_v2_stratified.py" \
        --experiment_id FT --checkpoint "$home/epoch_${epoch}.pth" \
        --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
        --manifest "$VAL/generation_manifest.jsonl" --output_dir "$dest" \
        --seed "$seed" --size 256 --batch 16 --workers 4
    done
    python "$PKG/selection_rule_v1.py" analyze \
      --manifest "$VAL/generation_manifest.jsonl" \
      --metrics-root "$home" --output "$home/selection_comparison.json"
  done
done
python "$PKG/compare_arms.py" --root "$OUT" --output "$OUT/paired_comparison.json"
