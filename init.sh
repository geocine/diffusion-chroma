#!/bin/bash

# Check if essential files are already present
if [ -f "run.sh" ] && [ -f "train.py" ]; then
    echo "Essential files already exist. Skipping clone operation."
else
    # Save our custom scripts first
    BACKUP_DIR=$(mktemp -d)
    if [ -f "init.sh" ]; then
        cp init.sh "$BACKUP_DIR/"
    fi

    # Create a temporary directory for cloning
    TEMP_REPO_DIR=$(mktemp -d)
    echo "Cloning repository to temporary directory: $TEMP_REPO_DIR"

    # Clone the repository with submodules to the temporary directory
    git clone --recurse-submodules https://github.com/geocine/diffusion-chroma "$TEMP_REPO_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone the repository."
        rm -rf "$TEMP_REPO_DIR"
        exit 1
    fi

    # Copy files from temporary directory to current directory, excluding our custom files
    echo "Copying files to current directory..."
    for item in $(ls -A "$TEMP_REPO_DIR"); do
        # Skip . and .. and our custom files
        if [[ "$item" != "." && "$item" != ".." && "$item" != "init.sh" ]]; then
            # Use -rf to handle directories and force overwrite
            cp -rf "$TEMP_REPO_DIR/$item" .
        fi
    done

    # Restore our backed up custom files
    if [ -f "$BACKUP_DIR/init.sh" ]; then
        cp "$BACKUP_DIR/init.sh" init.sh
        chmod +x init.sh
    fi

    # Clean up the temporary directory
    rm -rf "$TEMP_REPO_DIR"
    rm -rf "$BACKUP_DIR"
fi

# Make sure run.sh is executable
chmod +x run.sh

# Install PyTorch with CUDA support and other dependencies
echo "Installing dependencies..."
export PYTHONWARNINGS=ignore
export PIP_ROOT_USER_ACTION=ignore
pip install ninja
pip install torch torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu126 && pip install -r requirements.txt

# Make the download script executable and run it
if [ -f "download.sh" ]; then
    echo "Running download script..."
    chmod +x download.sh && ./download.sh
else
    echo "Error: download.sh not found. Installation might be corrupted."
    exit 1
fi

echo "Initialization complete!" 