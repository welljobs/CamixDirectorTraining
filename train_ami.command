#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
[[ -d data/ami_pilot/prepared ]] || { echo '请先运行 download_ami.command'; exit 1; }
RUN="runs/ami_$(date +%Y%m%d_%H%M%S)"
echo '弱监督实验，不是正式通用导播模型；不会操作 Camix。'
.venv/bin/python -m camix_director.train --data data/ami_pilot/prepared --output "$RUN" --config examples/pilot-config.json --epochs "${EPOCHS:-30}" --sequence-steps 16 --burn-in 4 --batch-size 2 --allow-weak-supervision --balance-actions
printf '%s\n' "$RUN" > runs/LATEST_AMI_RUN.txt
.venv/bin/python -m camix_director.evaluate --checkpoint "$RUN/best.pt" --data data/ami_pilot/prepared --output "$RUN/evaluation" --allow-weak-supervision
echo "完成：$ROOT/$RUN"
