#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
[[ -x .venv/bin/python ]] || { echo "请先运行 setup.command"; exit 1; }
echo "AMI 原分辨率720x576，不是720p/1080p。CC BY 4.0；标签为转写弱监督，不是导演真值。"
read -r -p "已阅读许可并保留出处，继续吗？[y/N] " answer
[[ "$answer" == y || "$answer" == Y ]] || exit 0
exec .venv/bin/python -u -m camix_director.ami_originals prepare --root data/ami_originals_576 --accept-ami-cc-by "$@"
