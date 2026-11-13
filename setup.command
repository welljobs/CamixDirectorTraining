#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  PY="$(command -v python3.11 || command -v python3.12 || true)"
  if [[ -z "$PY" ]]; then echo '请先安装 Python 3.11；不要使用 Python 3.14。'; exit 1; fi
  "$PY" -m venv .venv
fi
if .venv/bin/python -c 'import torch,numpy,av,pytest,coremltools; assert torch.__version__.startswith("2.7.")' >/dev/null 2>&1; then
  .venv/bin/python -m pip check
  echo '训练环境已可用。'
else
  .venv/bin/python bootstrap.py
fi
