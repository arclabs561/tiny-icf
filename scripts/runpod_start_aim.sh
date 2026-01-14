#!/bin/bash
# Start Aim UI on RunPod (for web terminal)

cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"

# Check if Aim is installed
if ! command -v aim &> /dev/null; then
    echo "Installing Aim..."
    uv pip install aim
fi

# Start Aim UI
echo "Starting Aim UI..."
echo "Access at: http://localhost:43800"
echo "For SSH tunnel: ssh -L 43800:localhost:43800 ..."
echo ""

aim up --host 0.0.0.0 --port 43800

