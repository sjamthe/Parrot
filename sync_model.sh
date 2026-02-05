#!/bin/bash

# Configuration
RUNPOD_HOST="194.68.245.62"
RUNPOD_PORT="22197"
SSH_KEY="$HOME/.ssh/id_ed25519"
MODEL_FILE="parrot_qwen_weights.pt"
REMOTE_PATH="/workspace/Parrot/$MODEL_FILE"
LOCAL_DIR="$HOME/Documents/GithubRepos/Parrot/models"
INTERVAL=540  # 9 minutes in seconds

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

echo "Starting model sync (every $INTERVAL seconds)..."
echo "Remote: $RUNPOD_HOST:$REMOTE_PATH"
echo "Local: $LOCAL_DIR"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Syncing model..."
    
    # Sync the main weights file
    scp -P "$RUNPOD_PORT" -i "$SSH_KEY" \
        "root@$RUNPOD_HOST:$REMOTE_PATH" \
        "$LOCAL_DIR/"
    
    if [ $? -eq 0 ]; then
        SIZE=$(ls -lh "$LOCAL_DIR/$MODEL_FILE" | awk '{print $5}')
        echo "[$TIMESTAMP] ✓ Synced successfully ($SIZE)"
        uv run test_qwen.py && play test_qwen_output.wav
    else
        echo "[$TIMESTAMP] ✗ Sync failed"
    fi
    
    # Also sync periodic checkpoints if they exist
    scp -P "$RUNPOD_PORT" -i "$SSH_KEY" \
        "root@$RUNPOD_HOST:/workspace/your-moshi-project/parrot_qwen_epoch_*.pt" \
        "$LOCAL_DIR/" 2>/dev/null
    
    echo "[$TIMESTAMP] Waiting $INTERVAL seconds..."
    sleep "$INTERVAL"
done
