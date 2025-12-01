# Training Guide

## Current Training Status

Training is running in the background with curriculum learning and all augmentations enabled.

## Monitor Training

### Quick Status Check
```bash
# Check if training is running
ps aux | grep "[p]ython.*train_curriculum"

# View recent progress
tail -20 training.log

# Analyze training progress
python scripts/analyze_training_progress.py --log training.log
```

### Continuous Monitoring
```bash
# Monitor script (updates every 10 seconds)
./scripts/monitor_training.sh training.log 10

# Or wait for completion and auto-validate
./scripts/wait_for_training.sh
```

## Training Configuration

**Current Run**:
- Data: 400k words, 2.2B tokens (multilingual)
- Typo Corpus: 82k real typo pairs
- Emoji Frequencies: 62 emojis/emoticons
- Epochs: 30
- Batch Size: 128
- Learning Rate: 0.001
- Curriculum: 5 stages, 3 warmup epochs
- Augmentation: Real typos + symbols + emojis + multilingual

**Model**:
- Architecture: UniversalICF (byte-level CNN)
- Parameters: 33,193 (< 50k constraint)
- Output: `models/model_curriculum_final.pt`

## After Training Completes

### Automatic Validation
```bash
# Wait for training and auto-run validation
./scripts/wait_for_training.sh
```

### Manual Validation
```bash
# Full validation suite
./scripts/post_training_validation.sh

# Or individual steps:
python scripts/validate_trained_model.py \
    --model models/model_curriculum_final.pt \
    --data data/multilingual/multilingual_combined.csv

python -m tiny_icf.predict \
    --model models/model_curriculum_final.pt \
    --words "the apple xylophone qzxbjk café привет 😀"
```

## Expected Results

### Training Metrics
- **Train Loss**: Should decrease to < 0.01
- **Val Loss**: Should decrease to < 0.02
- **Best Model**: Saved when validation loss improves

### Validation Targets
- **Jabberwocky Protocol**: 5/5 tests pass
- **Correlation**: Spearman > 0.8 with ground truth
- **Edge Cases**: Typos, multilingual, emojis all work

### Performance
- **Inference Speed**: < 5 ms/word (currently 1.08 ms ✓)
- **Throughput**: > 100 words/sec (currently 929 ✓)
- **Model Size**: < 50k params (33k ✓)

## Troubleshooting

### Training Stuck
```bash
# Check process
ps aux | grep train_curriculum

# Check log for errors
tail -50 training.log | grep -i error

# Restart if needed (will resume from checkpoint)
python -m tiny_icf.train_curriculum [args...]
```

### Model Not Learning
- Check if loss is decreasing
- Verify data is loading correctly
- Check augmentation is working
- May need more epochs or learning rate adjustment

### Validation Failing
- Model may need more training
- Check if ICF range is expanding (should be 0.0-1.0)
- Verify ground truth data quality

## Files

- **Training Log**: `training.log`
- **Model**: `models/model_curriculum_final.pt`
- **Scripts**: `scripts/analyze_training_progress.py`, `scripts/post_training_validation.sh`

