#!/usr/bin/env python3
"""Wrapper that launches the audio-to-audio translator server on port 8741.

The supported entry point is run.sh, which uses the project venv (Python 3.12).
This file is a convenience fallback for `python server.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

from backend.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8741, log_level="info")
