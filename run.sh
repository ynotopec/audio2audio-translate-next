#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TTS_BACKEND="${TTS_BACKEND:-qwen}"
echo "TTS_BACKEND=$TTS_BACKEND  (change with TTS_BACKEND=omnivoce ./run.sh)"

env -u PYTHONPATH \
  HOME="$HOME" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  CUDA_VISIBLE_DEVICES=0 \
  TTS_BACKEND="$TTS_BACKEND" \
  "$HOME/venv/audio2audio-translate-next-py312/bin/python3" -m uvicorn backend.main:app --host 0.0.0.0 --port 8741 --log-level info