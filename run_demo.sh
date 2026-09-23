#!/bin/bash
# 1-Click Startup Script for Skin AI Web Demo

echo "============================================================"
echo " Starting Skin Condition Classification System Demo..."
echo "============================================================"

# Navigate to project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Free up port 7860 from any old process
echo "Freeing port 7860..."
fuser -k 7860/tcp >/dev/null 2>&1 || true
sleep 1

# Step 2: Launch Web App directly
echo "Starting web server at http://127.0.0.1:7860..."
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/gui.py"
