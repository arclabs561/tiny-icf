# Justfile for ICF training automation
# Run with: just <command>

# RunPod connection details
# NOTE: Update these if pod IP/port changes
# Get from: RunPod console → Pod → SSH connection info
runpod_host := "root@213.173.111.79"
runpod_port := "37707"
runpod_key := "~/.ssh/id_ed25519"
runpod_path := "/root/idf-est"

# Default recipe
default:
    @just --list

# Setup ephemeral pod (clean, install dependencies)
setup-ephemeral:
    #!/usr/bin/env bash
    echo "Syncing code to RunPod..."
    just sync
    echo "Setting up ephemeral pod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} "cd {{runpod_path}} && bash scripts/runpod_setup_ephemeral.sh"
    echo "✓ Setup complete"

# Train NeuralNDCG for 100 epochs (ephemeral pod setup)
train-neural-ndcg-100ep:
    #!/usr/bin/env bash
    echo "Syncing code to RunPod..."
    just sync
    echo "Starting 100-epoch NeuralNDCG training..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} "cd {{runpod_path}} && bash scripts/runpod_train_neural_ndcg_100ep.sh"
    echo "✓ Training started. Monitor with: just monitor-neural-ndcg"

# Monitor NeuralNDCG training
monitor-neural-ndcg:
    #!/usr/bin/env bash
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} "cd {{runpod_path}} && echo '=== NeuralNDCG Training Status ===' && if [ -f logs/neural_ndcg_100ep.pid ]; then PID=\$(cat logs/neural_ndcg_100ep.pid) && ps -p \$PID > /dev/null 2>&1 && echo 'Running (PID: '\$PID')' || echo 'Not running'; else echo 'No PID file'; fi && echo '' && echo '=== GPU Usage ===' && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader && echo '' && echo '=== Latest Log (last 50 lines) ===' && tail -50 logs/neural_ndcg_100ep.log 2>/dev/null || echo 'No log file yet'"

# Sync code to RunPod
sync:
    #!/usr/bin/env bash
    echo "Syncing code to RunPod..."
    rsync -avz -e "ssh -p {{runpod_port}} -i {{runpod_key}}" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='models/*.pt' \
        --exclude='*.log' \
        --exclude='.env' \
        --exclude='ablation_results.json' \
        --exclude='training_history.json' \
        . {{runpod_host}}:{{runpod_path}}/
    echo "✓ Sync complete"

# Setup RunPod environment
setup:
    #!/usr/bin/env bash
    echo "Syncing setup script..."
    rsync -avz -e "ssh -p {{runpod_port}} -i {{runpod_key}}" \
        scripts/runpod_setup.sh {{runpod_host}}:{{runpod_path}}/scripts/
    echo "Running setup on RunPod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} \
        "chmod +x {{runpod_path}}/scripts/runpod_setup.sh && {{runpod_path}}/scripts/runpod_setup.sh"

# Sync and setup in one command
deploy: sync setup
    @echo "✓ Deployment complete"

# Run ablation study on RunPod (oneshot background)
ablation *args:
    #!/usr/bin/env bash
    echo "Syncing ablation scripts..."
    rsync -avz -e "ssh -p {{runpod_port}} -i {{runpod_key}}" \
        scripts/runpod_ablation*.sh scripts/runpod_monitor.sh {{runpod_host}}:{{runpod_path}}/scripts/
    echo "Starting ablation study in background on RunPod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} \
        "chmod +x {{runpod_path}}/scripts/runpod_ablation*.sh {{runpod_path}}/scripts/runpod_monitor.sh && {{runpod_path}}/scripts/runpod_ablation.sh {{args}}"
    echo "✓ Ablation study started in background"
    echo "Monitor with: just monitor"
    echo "With Aim: just ablation --aim"

# Run ablation with Aim tracking
ablation-aim:
    @just ablation --aim

# Monitor ablation study progress
monitor:
    #!/usr/bin/env bash
    echo "Checking ablation study status..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} \
        "cd {{runpod_path}} && bash scripts/runpod_monitor.sh"

# Quick training test (5 epochs)
quick-train *args:
    #!/usr/bin/env bash
    echo "Syncing quick train script..."
    rsync -avz -e "ssh -p {{runpod_port}} -i {{runpod_key}}" \
        scripts/runpod_quick_train.sh {{runpod_host}}:{{runpod_path}}/scripts/
    echo "Running quick training test on RunPod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} \
        "chmod +x {{runpod_path}}/scripts/runpod_quick_train.sh && {{runpod_path}}/scripts/runpod_quick_train.sh {{args}}"
    echo "✓ Quick test complete"

# Train with differentiable sorting
train-diffsort *args:
    #!/usr/bin/env bash
    echo "Syncing training script..."
    rsync -avz -e "ssh -p {{runpod_port}} -i {{runpod_key}}" \
        scripts/runpod_train_diffsort.sh {{runpod_host}}:{{runpod_path}}/scripts/
    echo "Training with differentiable sorting on RunPod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} \
        "chmod +x {{runpod_path}}/scripts/runpod_train_diffsort.sh && {{runpod_path}}/scripts/runpod_train_diffsort.sh {{args}}"
    echo "✓ Training complete"

# Train with listwise loss
train-listwise *args:
    #!/usr/bin/env bash
    echo "Training with listwise loss on RunPod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} bash -c "
        cd {{runpod_path}}
        export PATH=\"\$HOME/.cargo/bin:\$PATH\"
        uv run scripts/train_listwise.py \
            --data data/word_frequency.csv \
            --epochs 50 \
            --batch-size 64 \
            --listwise-method lambdarank \
            --output models/model_listwise.pt \
            {{args}} \
            2>&1 | tee train_listwise.log
    "
    echo "✓ Training complete"

# Train with best practices
train-best *args:
    #!/usr/bin/env bash
    echo "Training with best practices on RunPod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} bash -c "
        cd {{runpod_path}}
        export PATH=\"\$HOME/.cargo/bin:\$PATH\"
        uv run scripts/train_best_practices.py \
            --data data/word_frequency.csv \
            --epochs 100 \
            --batch-size 64 \
            --output models/model_best.pt \
            --early-stop \
            --early-stop-patience 15 \
            {{args}} \
            2>&1 | tee train_best.log
    "
    echo "✓ Training complete"

# Download results from RunPod
download:
    #!/usr/bin/env bash
    echo "Downloading results from RunPod..."
    mkdir -p models results logs
    scp -P {{runpod_port}} -i {{runpod_key}} \
        {{runpod_host}}:{{runpod_path}}/models/*.pt models/ 2>/dev/null || echo "No models to download"
    scp -P {{runpod_port}} -i {{runpod_key}} \
        {{runpod_host}}:{{runpod_path}}/*.json results/ 2>/dev/null || echo "No JSON results to download"
    scp -P {{runpod_port}} -i {{runpod_key}} \
        {{runpod_host}}:{{runpod_path}}/*.log logs/ 2>/dev/null || echo "No logs to download"
    echo "✓ Download complete"

# Watch training log in real-time
watch *logfile:
    #!/usr/bin/env bash
    echo "Watching {{logfile}} on RunPod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} \
        "cd {{runpod_path}} && tail -f {{logfile}}"

# Check RunPod status
status:
    #!/usr/bin/env bash
    echo "Checking RunPod status..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} bash -c "
        echo '=== System Info ==='
        echo \"Python: \$(python3 --version)\"
        echo \"UV: \$(uv --version 2>/dev/null || echo 'not installed')\"
        echo ''
        echo '=== GPU Info ==='
        nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'No GPU info'
        echo ''
        echo '=== Project Status ==='
        cd {{runpod_path}} 2>/dev/null && {
            echo \"Directory: \$(pwd)\"
            echo \"Data file: \$(test -f data/word_frequency.csv && echo '✓ Found' || echo '✗ Missing')\"
            echo \"Models: \$(ls -1 models/*.pt 2>/dev/null | wc -l) files\"
            echo \"Logs: \$(ls -1 *.log 2>/dev/null | wc -l) files\"
        } || echo 'Project not found'
    "

# Run all experiments (deploy, ablation, train)
all:
    @just deploy
    @just ablation
    @just train-diffsort
    @just download

# Quick test connection
test-connection:
    #!/usr/bin/env bash
    echo "Testing RunPod connection..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} "echo '✓ Connection successful' && python3 --version"

# Start Aim UI for experiment tracking (local)
aim:
    #!/usr/bin/env bash
    echo "Starting Aim UI..."
    echo "Access at: http://127.0.0.1:43800"
    aim up

# Start Aim UI on RunPod (with port forwarding)
aim-remote:
    #!/usr/bin/env bash
    echo "Starting Aim UI on RunPod..."
    echo "Setting up SSH tunnel: localhost:43800 -> RunPod:43800"
    echo "Access at: http://127.0.0.1:43800"
    ssh -p {{runpod_port}} -i {{runpod_key}} -L 43800:localhost:43800 {{runpod_host}} \
        "cd {{runpod_path}} && aim up" &
    echo "Tunnel running in background. Press Ctrl+C to stop."
    wait

# Train with best practices and Aim tracking
train-best-aim *args:
    #!/usr/bin/env bash
    echo "Training with best practices + Aim tracking on RunPod..."
    ssh -p {{runpod_port}} -i {{runpod_key}} {{runpod_host}} bash -c "
        cd {{runpod_path}}
        export PATH=\"\$HOME/.cargo/bin:\$PATH\"
        uv run scripts/train_best_practices.py \
            --data data/word_frequency.csv \
            --epochs 100 \
            --batch-size 64 \
            --output models/model_best.pt \
            --early-stop \
            --early-stop-patience 15 \
            --aim \
            --aim-experiment \"runpod-experiments\" \
            {{args}} \
            2>&1 | tee train_best.log
    "
    echo "✓ Training complete. View results with: just aim"

# Build Typst documentation (PDF and HTML)
build-docs:
    #!/usr/bin/env bash
    echo "Building Typst documentation..."
    bash scripts/build_typst_docs.sh

# Install just if not present
install-just:
    #!/usr/bin/env bash
    if ! command -v just &> /dev/null; then
        echo "Installing just..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install just
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
        else
            echo "Please install just manually: https://github.com/casey/just"
        fi
    else
        echo "just is already installed"
    fi

