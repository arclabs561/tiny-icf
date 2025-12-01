# RunPod Setup Status

## ✅ Configuration Complete

- **API Key**: Extracted from `~/.cursor/mcp.json`
- **runpodctl**: Configured with API key
- **Pods Created**: 
  - `w4857gh85qldts` - RUNNING
  - `2ybkbqm9l3wwz1` - RUNNING

## Quick Start

### Get SSH Connection
```bash
runpodctl ssh connect w4857gh85qldts
```

### Upload Files
Use the SSH connection string from above, or use SCP:
```bash
# Get SSH host from runpodctl
SSH_HOST=$(runpodctl ssh connect w4857gh85qldts | grep -oE '[a-z0-9]+@[^ ]+')

# Upload project
scp -r . $SSH_HOST:/workspace/tiny-icf
```

### SSH and Train
```bash
# SSH into pod
runpodctl ssh connect w4857gh85qldts

# Inside pod:
cd /workspace/tiny-icf
uv pip install -e .
python -m tiny_icf.train_curriculum \
    --data data/multilingual/multilingual_combined.csv \
    --typo-corpus data/typos/github_typos.csv \
    --multilingual \
    --epochs 50 \
    --batch-size 128 \
    --output models/model_runpod.pt
```

## Using MCP Tools (Recommended)

Since RunPod MCP server is configured in Cursor, you can use MCP tools directly:

1. **Create/Manage Pods**: Use MCP RunPod tools
2. **Upload Files**: Use MCP file transfer
3. **Execute Commands**: Use MCP execution tools

## Cost

- **RTX 3090**: $0.220/hour
- **Estimated Training Time**: 2-4 hours
- **Estimated Cost**: $0.44-$0.88

## Cleanup

```bash
# Stop pod (keeps data)
runpodctl stop w4857gh85qldts

# Remove pod (deletes data)
runpodctl remove w4857gh85qldts
```

