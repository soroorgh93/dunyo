#!/usr/bin/env bash
# Dunyo launcher (macOS/Linux)
cd "$(dirname "$0")"
[ -x venv/bin/python ] || { python3 -m venv venv && venv/bin/pip install -r requirements.txt; }
exec venv/bin/python -m dunyo app "$@"
