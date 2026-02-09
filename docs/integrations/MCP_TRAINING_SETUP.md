# MCP RunPod Training Setup

## Current Status

✅ **Pods Available:**
- `w4857gh85qldts` - RUNNING (RTX 3090, $0.220/hr)
- `2ybkbqm9l3wwz1` - RUNNING (RTX 3090, $0.220/hr)

✅ **API Key**: Configured from `~/.cursor/mcp.json`

## MCP Commands to Execute

### 1. Upload Project Files

**Command:**
```
Upload the entire tiny-icf project directory to RunPod pod w4857gh85qldts at /workspace/tiny-icf
```

**What to upload:**
- All source code (`src/`)
- Training scripts
- Data files (`data/`)
- Configuration files (`pyproject.toml`, etc.)

### 2. Install Dependencies

**Command:**
```
Execute on RunPod pod w4857gh85qldts: cd /workspace/tiny-icf && uv pip install -e .
```

### 3. Start Training

**Command:**
```
Execute on RunPod pod w4857gh85qldts:
cd /workspace/tiny-icf && python -m tiny_icf.train_curriculum \
    --data data/multilingual/multilingual_combined.csv \
    --typo-corpus data/typos/github_typos.csv \
    --multilingual \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-3 \
    --curriculum-stages 5 \
    --output models/model_runpod.pt
```

### 4. Monitor Training

**Command:**
```
Show the last 50 lines of /workspace/tiny-icf/training.log from RunPod pod w4857gh85qldts
```

Or:
```
Monitor training progress on RunPod pod w4857gh85qldts by tailing /workspace/tiny-icf/training.log
```

## Training Configuration

- **Model**: UniversalICF (33k parameters, byte-level CNN)
- **Data**: 400k words, 2.2B tokens (multilingual)
- **Typo Corpus**: 82k real typo pairs
- **Epochs**: 50
- **Batch Size**: 128 (GPU optimized)
- **Learning Rate**: 1e-3
- **Curriculum**: 5 stages with progressive difficulty
- **Augmentation**: Real typos + multilingual + symbols + emojis

## Expected Results

- **Training Time**: 2-4 hours on RTX 3090
- **Cost**: ~$0.44-$0.88
- **Output**: `models/model_runpod.pt`
- **Validation**: Spearman correlation > 0.8

## After Training

### Download Model
```
Download /workspace/tiny-icf/models/model_runpod.pt from RunPod pod w4857gh85qldts to ./models/
```

### Validate Model
```
Execute on RunPod pod w4857gh85qldts:
cd /workspace/tiny-icf && python scripts/validate_trained_model.py \
    --model models/model_runpod.pt \
    --data data/multilingual/multilingual_combined.csv
```

## Cleanup

When done:
```
Stop RunPod pod w4857gh85qldts
```

Or to remove completely:
```
Remove RunPod pod w4857gh85qldts
```

