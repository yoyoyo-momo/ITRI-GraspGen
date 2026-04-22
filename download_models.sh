#!/bin/bash
set -e

# This script downloads all the necessary models.
# Run this script from the project root directory (ITRI-GraspGen).

PROJECT_ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$PROJECT_ROOT_DIR/.venv/bin/python"

SKIP_FOUNDATION_STEREO=0

for arg in "$@"; do
    case "$arg" in
        --skip-foundation-stereo)
            SKIP_FOUNDATION_STEREO=1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--skip-foundation-stereo]"
            exit 1
            ;;
    esac
done

MODELS_DIR="models"

# GraspGenModels
echo "Downloading GraspGenModels..."
if [ -d "$MODELS_DIR/GraspGenModels" ]; then
    echo "GraspGenModels already exists, skipping."
else
    git clone https://huggingface.co/adithyamurali/GraspGenModels "$MODELS_DIR/GraspGenModels"
fi

# SAM2Models
echo "Downloading SAM2Models..."
if [ -d "$MODELS_DIR/SAM2Models" ] && [ "$(ls -A $MODELS_DIR/SAM2Models)" ]; then
    echo "SAM2Models already exists and is not empty, skipping."
else
    mkdir -p "$MODELS_DIR/SAM2Models"
    cp Third_Party/sam2/checkpoints/download_ckpts.sh "$MODELS_DIR/SAM2Models/"
    (cd "$MODELS_DIR/SAM2Models" && bash ./download_ckpts.sh)
fi

# FoundationStereoModels
if [ "$SKIP_FOUNDATION_STEREO" -eq 1 ]; then
    echo "Skipping FoundationStereoModels download (--skip-foundation-stereo set)."
else
    echo "Downloading FoundationStereoModels..."
    if [ -d "$MODELS_DIR/FoundationStereoModels" ]; then
        echo "FoundationStereoModels already exists, skipping."
    else
        # gdown downloads to the current directory, so we execute it inside the models dir.
        # Prefer project .venv to avoid requiring global installs.
        if command -v gdown >/dev/null 2>&1; then
            (cd "$MODELS_DIR" && gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf)
        elif [ -x "$VENV_PYTHON" ]; then
            (cd "$MODELS_DIR" && "$VENV_PYTHON" -m gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf)
        elif command -v python3 >/dev/null 2>&1; then
            (cd "$MODELS_DIR" && python3 -m gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf)
        else
            echo "Error: gdown is not available. Install it in the venv with: .venv/bin/pip install gdown"
            exit 1
        fi
        mv "$MODELS_DIR/pretrained_models" "$MODELS_DIR/FoundationStereoModels"
    fi
fi

# GroundingDINOModels
echo "Downloading GroundingDINOModels..."
if [ -f "$MODELS_DIR/GroundingDINOModels/groundingdino_swint_ogc.pth" ]; then
    echo "GroundingDINO model already exists, skipping."
else
    mkdir -p "$MODELS_DIR/GroundingDINOModels"
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P "$MODELS_DIR/GroundingDINOModels"
fi


echo "All models downloaded successfully."
