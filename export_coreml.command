#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
[[ -f runs/LATEST_AMI_RUN.txt ]] || { echo '请先完成 train_ami.command'; exit 1; }
RUN="$(cat runs/LATEST_AMI_RUN.txt)"
OUT="$RUN/coreml_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m camix_director.export --checkpoint "$RUN/best.pt" --output "$OUT" --precision fp32 --parity-steps 12
