#!/bin/bash
# One-command training starter - copy and paste this
POD_ID="9lj0lizlogeftc"

echo "🚀 Starting training on $POD_ID..."
runpodctl ssh connect "$POD_ID" << 'SSH_COMMANDS'
cd /workspace/tiny-icf
bash start_mcp.sh
exit
SSH_COMMANDS

echo "✅ Command executed"

