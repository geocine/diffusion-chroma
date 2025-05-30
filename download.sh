#!/bin/bash

# Set the root directory as the current directory
ROOT_DIR="$(pwd)"

# Create models directory structure if it doesn't exist
mkdir -p "$ROOT_DIR/models/flux"
mkdir -p "$ROOT_DIR/models/chroma"  # Create directory for full model download

if [ -z "$HF_TOKEN" ]; then
    if [ ! -f "token" ]; then
        touch token
        echo "Token file created. Please add your Hugging Face token to the 'token' file and run the script again."
        exit 1
    elif [ ! -s "token" ]; then
        echo "Token file is empty. Please add your Hugging Face token to the 'token' file and run the script again."
        exit 1
    fi
    export HF_TOKEN=$(cat token)
else
    echo "Using HF_TOKEN from environment variable."
fi

pip install -U "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# https://github.com/huggingface/huggingface_hub/pull/2223
download_if_not_exists() {
    local repo=$1
    local file=$2
    local dest_dir=${3:-$ROOT_DIR}
    local filename=${4:-$file}
    local dest="$dest_dir/$filename"
    
    if [ ! -f "$dest" ]; then
        echo "Downloading $file..."
        huggingface-cli download "$repo" "$file" --local-dir "$dest_dir"
        
        if [ $? -ne 0 ]; then
            echo "Error downloading $file from $repo"
            return 1
        fi
        
        echo "Downloaded $file to $dest_dir"
    else
        echo "File $filename already exists at $dest_dir. Skipping download."
    fi
}

download_folder() {
    local repo=$1
    local folder=$2
    local dest_dir=$3
    
    echo "Downloading folder $folder from $repo to $dest_dir..."
    huggingface-cli download "$repo" --repo-type model --local-dir "$dest_dir" --include "$folder/**"
    
    if [ $? -ne 0 ]; then
        echo "Error downloading folder $folder from $repo"
        return 1
    fi
    
    echo "Downloaded folder $folder to $dest_dir"
}

download_full_model() {
    local repo=$1
    local dest_dir=$2
    
    echo "Downloading full model from $repo to $dest_dir..."
    huggingface-cli download "$repo" --repo-type model --local-dir "$dest_dir"
    
    if [ $? -ne 0 ]; then
        echo "Error downloading full model from $repo"
        return 1
    fi
    
    echo "Downloaded full model from $repo to $dest_dir"
}


# Download the full FLUX.1-schnell model
echo "Downloading the full FLUX.1-schnell model..."
download_full_model "black-forest-labs/FLUX.1-schnell" "$ROOT_DIR/models/flux"

# Download Chroma model weights
echo "Downloading Chroma model weights..."
download_if_not_exists "lodestones/Chroma" "chroma-unlocked-v32.safetensors" "$ROOT_DIR/models/chroma"

echo "All downloads completed. Models are stored in the '$ROOT_DIR/models' directory."
echo "Directory structure:"
echo "models/"
echo "├── flux/ (contains the full FLUX.1-schnell model)"
echo "└── chroma/"
echo "    └── chroma-unlocked-v32.safetensors" 