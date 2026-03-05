<div align="center">

<h2>
  <img src="assets/dynaavatar_logo_r.png" width="50" style="vertical-align: middle; margin-right: 10px;">
  DynaAvatar: Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image <br> (CVPR 2026)
</h2>

[**[Project Page]**](https://juhyeon-kwon.github.io/DynaAvatar.github.io/) | [**[Paper]**](#) | [**[arXiv]**](#)
</div>

---

## TODO
- [x] Release inference code
- [x] Release train code
- [ ] Release re-annotated datasets

---

## 🚀 Getting Started

### 1. Installation
To set up the environment, please refer to **[INSTALL.md](./INSTALL.md)** for step-by-step instructions.

> [!IMPORTANT]
> We performed the installation using the commands documented in `INSTALL.md`. Please ensure your **Torch** and **CUDA** versions are installed carefully to match your specific hardware environment.

### 2. Model & Data Preparation

Follow these steps to download the required models and configure the project paths.

#### **A. Download prior Models**
Download the LHM prior weights and extract them:
```bash
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar 
tar -xvf LHM_prior_model.tar

# Download pretrained LHM-500M
python download.py
```

#### **B. Update LHM Model Configuration**
Next, you must manually update the configuration file located at:  
`PATH/DynaAvatar_RELEASE/pretrained_models/Damo_XR_Lab/LHM-500M/step_060000/config.json`

Add these fields in the JSON file:
```json
{
    // ...

    "n_history_length": 15,
    "is_dynamic": true
}
```

#### **C. DynaAvatar Checkpoints**
Download the pretrained [DynaAvatar checkpoint](https://drive.google.com/drive/folders/1ypHIxlmAUUDRYIYZUNTTeoW3G84ge8hZ?usp=drive_link) and update the YAML configuration.
* **Config Path**: `PATH/DynaAvatar_RELEASE/configs/inference/human-lrm-500M.yaml`
* **Action**: Update the `saver.load_model` field to point to the absolute path of your downloaded checkpoint.

#### **D. Additional Assets**
* **Example Motion Sequences**: Download the [sample motion sequences](https://drive.google.com/drive/folders/1m7P2ErOKxp3JdcSTDWX2VjFwGTeZ6uhw?usp=drive_link) required for inference.
* **Voxel Grid**: Download the [volume voxel grid](https://drive.google.com/drive/folders/1KvNgPfwdyecUrKKVaowa4-Tdk6gy2m0o?usp=drive_link). Ensure the voxel grid files are placed in:  
  `PATH/DynaAvatar_RELEASE/pretrained_models/volume_voxel_grid`

---

## 💃 Inference 

### Usage Example
Here are two examples using different motion sequences (e.g., DNA-Rendering and 4D-DRESS datasets). 

#### **Example 1: DNA-Rendering sequence**
```bash
CUDA_VISIBLE_DEVICES=0 bash inference.sh LHM-500M \
    /data1/qw00n/DynaAvatar_RELEASE/assets/novel_subject \
    PATH/motion_seqs/DNA_Rendering/0124_03/smplx/smplx_params_smooth \
    None 0 500 15
```

#### **Example 2: 4D-DRESS sequence**
```bash
CUDA_VISIBLE_DEVICES=0 bash inference.sh LHM-500M \
    /data1/qw00n/DynaAvatar_RELEASE/assets/novel_subject \
    PATH/motion_seqs/4D-DRESS/00152_outer_16/smplx/smplx_params_smooth \
    None 0 500 30
```

### Arguments
```bash
bash inference.sh [MODEL_NAME] [SOURCE_IMAGE_DIR] [MOTION_PARAM_PATH] [BG_PATH] [START_FRAME] [MOTION_SIZE] [FPS]
```

* **MODEL_NAME**: Name of the backbone model (e.g., `LHM-500M`).
* **SOURCE_IMAGE_DIR**: Path to the folder containing source images.
* **MOTION_PARAM_PATH**: Path to the SMPL-X motion parameters.
* **MOTION_SIZE**: Maximum number of frames to render (e.g., `300`).
* **FPS**: Frames per second (e.g., `15` or `30`).

> 💡 **Note:** If you encounter VRAM issues, please adjust the `batch_size` in [`DynaAvatar_RELEASE/LHM/runners/infer/human_lrm.py`](https://github.com/juhyeon-kwon/DynaAvatar_RELEASE/blob/4f5f372a987deb3aa8d8f2ee2fb900361b1df9dd/LHM/runners/infer/human_lrm.py#L969C1-L970C1) at line **969**.
---

## 🕺 Training 
Training code is now available; a detailed guide is currently under preparation.

---

## Acknowledgement
This work is built upon several amazing open-source projects: [PERSONA](https://github.com/mks0601/PERSONA_RELEASE), [LHM](https://github.com/aigc3d/LHM), and others. We are grateful for their excellent contributions to the community.

---

## Citation
If you find our work helpful, please cite:
```bibtex

```