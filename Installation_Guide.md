# Installation Guide

## Prerequisite
- Ubuntu-22.04
- Nvidia-driver
- Python 3.12
- CUDA-13.0
- uv
- git-lfs

#### uv Installation
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
#### git-lfs Installation
```bash
sudo apt install git-lfs
git lfs install
```

## Installation

## Compatibility Matrix (Recommended)
- Python: 3.12.x
- CUDA Toolkit: 13.0
- PyTorch: 2.11 + cu130
- TorchVision: 0.26 + cu130
- TorchAudio: 2.11 + cu130

#### Load CUDA-13.0
- If you're on personal PC, and you do have cuda-13.0 installed at **/usr/local/cuda-13.0**:
    ```bash
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH=/usr/local/cuda-13.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    ```

#### Clone this repo
```
git clone git@github.com:Pxter7777/ITRI-GraspGen.git
```
#### Update submodules
```bash
git submodule update --init --recursive
```

#### Create Python 3.12 virtual environment
```bash
uv venv --python 3.12
source .venv/bin/activate
```

#### Install dependencies
```bash
uv sync
```
- This could take a while

#### Build local pointnet2 extension
```bash
source .venv/bin/activate
cd Third_Party/GraspGen/pointnet2_ops
python -m pip install --no-build-isolation . -v
cd ../../..
```

#### **Note on `groundingdino/version.py`:**
After running `uv sync`, you might notice an untracked file: `Third_Party/GroundingDINO/groundingdino/version.py`. This file is generated during the installation process of the `groundingdino` submodule. To prevent this file from cluttering your `git status`, you can add it to your local Git exclude list:

```bash
echo "groundingdino/version.py" >> .git/modules/Third_Party/GroundingDINO/info/exclude
```
This command only needs to be run once. It tells Git to ignore the file locally without modifying the submodule's `.gitignore` or the main project's `.gitignore`. This change is local to your repository and will not be committed.

## Install ZED SDK
#### Download
- Go visit [ZED SDK official site](https://www.stereolabs.com/en-fr/developers/release)
- download **ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_\<version\>.zstd.run**
    - Or just copy from my external disk.
#### Install (need sudo)
```bash
./ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_<version>.zstd.run
```

```
To continue you have to accept the EULA. Accept  [Y/n] ? y

ZED SDK will be installed in: /usr/local/zed
[sudo] password for j300: <your_password>

Installing TensorRT 10.9, mandatory dependencies to use the ZED SDK
Install samples (recommended) [Y/n] ? y
Installation path: /usr/local/zed/samples/


Do you want to install the Python API (recommended) [Y/n] ? n
```
- Notice:
    - We install it into project venv manually after SDK installation.
```
continue...

Do you want to download and optimize the NEURAL Depth models now? These will be required at runtime and will be processed then if not done now, which will extend startup time on first use. [Y/n] ? y

Do you want to run the ZED Diagnostic to download all AI models [Y/n] ? n
```

#### Install ZED Python API into the project venv
```bash
source .venv/bin/activate
python /usr/local/zed/get_python_api.py
```

## Download Models
```bash
bash download_models.sh
```

## IsaacSim related installation see [here](./isaac-sim2real/README.md)