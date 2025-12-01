#!/bin/bash
# Use expect to automate SSH and start training
POD_ID="${1:-9lj0lizlogeftc}"

echo "🚀 Starting training via automated SSH..."

# Get SSH command
SSH_CMD=$(runpodctl ssh connect "$POD_ID" 2>&1 | grep -oE 'ssh [^ ]+@[^ ]+' | head -1)

if [ -z "$SSH_CMD" ]; then
    echo "❌ Could not get SSH command"
    exit 1
fi

echo "   Using: $SSH_CMD"

# Use expect to automate SSH
expect << EOF
set timeout 30
spawn $SSH_CMD "cd /workspace/tiny-icf && bash start_mcp.sh && echo 'Training started'"
expect {
    "password:" { exit 1 }
    "Password:" { exit 1 }
    "Are you sure" { send "yes\r"; exp_continue }
    "yes/no" { send "yes\r"; exp_continue }
    eof
}
catch wait result
exit [lindex \$result 3]
EOF

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training started!"
else
    echo "⚠️  Expect had issues (exit code: $EXIT_CODE)"
    echo "   Training may still have started - check pod"
fi

