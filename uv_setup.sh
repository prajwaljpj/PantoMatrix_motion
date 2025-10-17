#!/usr/bin/env bash
set -e

# Update package lists
sudo apt-get update

# Install system dependencies
sudo apt-get install -y bzip2 tmux git git-lfs libglu1-mesa-dev

# Create and activate virtual environment
uv venv -p python3.9
source .venv/bin/activate

# Install Python dependencies
uv pip install --extra-index-url https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt200/download.html -r ./uv_requirements.txt

# Git LFS and clone repository
git lfs install
git clone https://huggingface.co/H-Liu1997/emage_evaltools
