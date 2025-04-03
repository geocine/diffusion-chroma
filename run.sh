#!/bin/bash

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Error: No configuration file provided."
    echo "Usage: ./run.sh <config_file.toml> [--clear]"
    exit 1
fi

CONFIG_FILE=$1
CLEAR_FLAG=""

# Check for --clear flag
if [[ "$@" == *"--clear"* ]]; then
    CLEAR_FLAG="--clear"
fi

# Run the caption script
echo "Running caption.py..."
python caption.py $CLEAR_FLAG

# Check if outputs directory exists and contains checkpoint folders
RESUME_OPTION=""
OUTPUT_DIR="outputs"
if [ -d "$OUTPUT_DIR" ]; then
    # Get list of checkpoint folders (assuming they're named with date pattern)
    CHECKPOINT_FOLDERS=$(ls -d $OUTPUT_DIR/*_*-*-* 2>/dev/null || true)
    
    if [ ! -z "$CHECKPOINT_FOLDERS" ]; then
        # Find the most recent checkpoint folder (sort by name, which works for date-formatted folders)
        LATEST_CHECKPOINT=$(echo "$CHECKPOINT_FOLDERS" | sort -r | head -n 1 | xargs basename)
        
        if [ ! -z "$LATEST_CHECKPOINT" ]; then
            echo "Found checkpoint: $LATEST_CHECKPOINT"
            RESUME_OPTION="--resume_from_checkpoint $LATEST_CHECKPOINT"
        fi
    else
        echo "No checkpoint folders found in $OUTPUT_DIR"
    fi
else
    echo "No outputs directory found"
fi

# Run the training with deepspeed
echo "Starting training with $CONFIG_FILE..."
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config $CONFIG_FILE $RESUME_OPTION

echo "Training complete!" 