#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo '请先运行 setup.command 创建本机训练环境。'
  exit 1
fi
exec .venv/bin/python -u -m camix_director.web "$@"
