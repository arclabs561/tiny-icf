# Using MCP RunPod Tools in Cursor

## Available MCP Tools

Since RunPod MCP server is configured in Cursor, you can use natural language commands to:

1. **List Pods**: "List all my RunPod pods"
2. **Create Pod**: "Create a RunPod pod with NVIDIA RTX 3090 for training"
3. **Upload Files**: "Upload the tiny-icf project to pod [pod-id]"
4. **Execute Commands**: "Run training on pod [pod-id]"
5. **Monitor Training**: "Check training logs on pod [pod-id]"

## Current Pods

- `w4857gh85qldts` - RUNNING (RTX 3090)
- `2ybkbqm9l3wwz1` - RUNNING (RTX 3090)

## Training Setup via MCP

### Step 1: Upload Project Files
```
Upload the tiny-icf project directory to pod w4857gh85qldts at /workspace/tiny-icf
```

### Step 2: Install Dependencies
```
Execute on pod w4857gh85qldts: cd /workspace/tiny-icf && uv pip install -e .
```

### Step 3: Start Training
```
Execute on pod w4857gh85qldts:
cd /workspace/tiny-icf && python -m tiny_icf.train_curriculum \
    --data data/multilingual/multilingual_combined.csv \
    --typo-corpus data/typos/github_typos.csv \
    --multilingual \
    --epochs 50 \
    --batch-size 128 \
    --output models/model_runpod.pt
```

### Step 4: Monitor Training
```
Show logs from pod w4857gh85qldts at /workspace/tiny-icf/training.log
```

## Training Configuration

- **Data**: `data/multilingual/multilingual_combined.csv` (400k words, 2.2B tokens)
- **Typo Corpus**: `data/typos/github_typos.csv` (82k real typo pairs)
- **Epochs**: 50
- **Batch Size**: 128 (GPU optimized)
- **Learning Rate**: 1e-3
- **Output**: `models/model_runpod.pt`

## Expected Training Time

- **On RTX 3090**: ~2-4 hours for 50 epochs
- **Cost**: ~$0.44-$0.88 at $0.220/hour

