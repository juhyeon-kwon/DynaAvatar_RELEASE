# Installation
  Our experiments were conducted on an **NVIDIA RTX pro 6000 (Blackwell)** server. However, it is also compatible with other environments such as **A100 clusters**, provided that the PyTorch and CUDA versions are correctly matched.
## Requirements
- Linux
- Python 3.10

## 0. Create Conda environment
  ```bash
  # Revise the yml file to match your environment
  conda env create -f base.yml
  conda activate dynaavatar_release
  ```

## 1. Install pytorch
  ```bash
  # Install Torch and CUDA versions matching your environment
  pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
  ```

## 2. Install xformers
  ```bash
  # cuda 11.8
  pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118

  # cuda 12.1
  pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

  # For Blackwell architecture, build from source
  # Refer to: https://github.com/Dao-AILab/flash-attention/issues/1763#issuecomment-3418680836
  pip install ninja
  export TORCH_CUDA_ARCH_LIST="12.0"
  pip install --no-build-isolation --pre -v -U git+https://github.com/facebookresearch/xformers.git@fde5a2fb46e3f83d73e2974a4d12caf526a4203e
  ```

## 3. Install base dependencies
  ```bash
  pip install -r requirements.txt
  pip install chumpy --no-build-isolation

  # install from source code to avoid the conflict with torchvision
  pip uninstall basicsr
  pip install git+https://github.com/XPixelGroup/BasicSR
  ```

## 4. Install SAM2 lib. We use the modified version following LHM
```bash
pip install --no-build-isolation git+https://github.com/hitsz-zuoqi/sam2/

# or
git clone --recursive https://github.com/hitsz-zuoqi/sam2
pip install ./sam2
```

## 5. Install 3DGS python-bings
```bash
# we use the version modified by Jiaxiang Tang, thanks for this great job!
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install --no-build-isolation ./diff-gaussian-rasterization

# simple-knn
git clone https://github.com/camenduru/simple-knn.git
pip install --no-build-isolation ./simple-knn
```

## 6. Install [Pytorch3D](https://github.com/facebookresearch/pytorch3d)
```bash
  # We used the following commands
  pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" 
```

## 7. Install [LightGlue](https://github.com/cvg/LightGlue)



