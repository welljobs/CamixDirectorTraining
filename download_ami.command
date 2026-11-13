#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
[[ -x .venv/bin/python ]] || { echo '请先运行 setup.command'; exit 1; }
echo '仅下载三场 AMI 会议中两路相机/音轨的约 45 秒片段。'
echo '来源：https://groups.inf.ed.ac.uk/ami/corpus/；信号及转写：CC BY 4.0。'
echo '保留出处/修改声明；标签是说话人时间生成的弱监督，不是专业导演真值。'
read -r -p '已阅读来源许可，继续吗？[y/N] ' answer
[[ "$answer" == y || "$answer" == Y ]] || exit 0
.venv/bin/python -m camix_director.ami --root data/ami_pilot --config examples/pilot-config.json --seconds 45 --accept-ami-cc-by
