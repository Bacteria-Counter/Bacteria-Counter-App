#!/bin/bash
# Starts the local inference server used by Bacteria-Counter-App.
# Run this once before opening the app, and leave it running in this terminal.
set -e
cd "$(dirname "$0")"
.venv/bin/python server.py
