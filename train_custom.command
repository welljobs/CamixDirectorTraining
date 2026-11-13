#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
if [[ $# -lt 2 ]]; then
 echo '用法：bash train_custom.command /绝对路径/prepared /绝对路径/config.json [输出目录]'
 echo '数据必须是经过审核的真实同步数据；不会自动降级使用合成数据或弱标签。'
 exit 2
fi
RUN="${3:-runs/custom_$(date +%Y%m%d_%H%M%S)}"
.venv/bin/python -m camix_director.train --data "$1" --config "$2" --output "$RUN" --epochs "${EPOCHS:-20}" --sequence-steps "${SEQUENCE_STEPS:-64}" --burn-in "${BURN_IN:-16}" --batch-size "${BATCH_SIZE:-2}"
