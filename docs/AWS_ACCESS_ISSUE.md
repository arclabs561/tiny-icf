# AWS Instance Access Issue

## Problem
AWS GPU instances are running but not accessible via SSH:
- Connection timeout
- IP extraction failing (showing "-" or "None")
- Security groups may not allow SSH from current IP

## Instances Status
8 instances running (g4dn.xlarge):
- i-04da119c071468ece: 3.236.98.2 (spot)
- i-0f3d87ed0ca292eb2: 98.92.241.131 (on-demand)
- i-0baa3f416a744431f: 3.236.149.31 (spot)
- i-089543631c21cc43f: 44.212.92.124 (spot)
- i-0b6ff045f9f3ea3b0: 18.232.56.23 (spot)
- i-04d3faa98b8bc6332: 44.222.170.222 (spot)
- i-0e36819e338793d00: 44.222.116.19 (spot)
- i-04bdff25262f91d2f: 3.215.186.237 (spot)

## Solutions

### Option 1: Fix Security Groups
```bash
# Get your current IP
MY_IP=$(curl -s https://checkip.amazonaws.com)

# Get security group ID for instances
SG_ID=$(aws ec2 describe-instances --instance-ids i-04da119c071468ece --region us-east-1 --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)

# Add SSH rule
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr $MY_IP/32 \
    --region us-east-1
```

### Option 2: Use AWS Systems Manager Session Manager
```bash
# Install Session Manager plugin
# Then connect without SSH:
aws ssm start-session --target i-04da119c071468ece --region us-east-1
```

### Option 3: Launch New Instance with Proper Access
```bash
# Use scale_gpu_training.sh with proper security group
bash scripts/scale_gpu_training.sh launch g4dn.xlarge
```

### Option 4: Run Locally (CPU Fallback)
Training script works on CPU, just slower:
```bash
uv run scripts/train_flexible_opportunistic.py \
    --experiments residual_optimal residual_wide residual_deep residual_balanced \
    --data data/word_frequency.csv \
    --output-dir models
```

## Current Status
- Training launched locally (CPU fallback)
- AWS instances remain inaccessible
- Need to fix security groups or use alternative access method

