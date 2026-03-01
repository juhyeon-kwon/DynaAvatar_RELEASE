# 💃 DynaAvatar: Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image

## CVPR 2026

### [[Project Page]](#) | [[Paper]](#) | [[arXiv]](#) | [[Poster]](#) | [Video](#)

**This is the official PyTorch implementation of the approach described in the following paper:**
> **DynaAvatar: Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image**\
> [Joohyun Kwon*], [Geonhee Sim*], and [Gyeongsik Moon†]\
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026

---

## Abstract
![overall_framework](assets/intro_compare.jpg)
Existing single-image 3D human avatar methods primarily rely on rigid joint transformations, limiting their ability to model realistic cloth dynamics. We present DynaAvatar, a zero-shot framework that reconstructs animatable 3D human avatars with motion-dependent cloth dynamics from a single image. Trained on large-scale multi-person motion datasets, DynaAvatar employs a Transformer-based feed-forward architecture that directly predicts dynamic 3D Gaussian deformations without subject-specific optimization. To overcome the scarcity of dynamic captures, we introduce a static-to-dynamic knowledge transfer strategy: a Transformer pretrained on large-scale static captures provides strong geometric and appearance priors, which are efficiently adapted to motion-dependent deformations through lightweight LoRA fine-tuning on dynamic captures. We further propose the DynaFlow loss, an optical flow–guided objective that provides reliable motion-direction geometric cues for cloth dynamics in rendered space. Finally, we reannotate the missing or noisy SMPL-X fittings in existing dynamic capture datasets, as most public dynamic capture datasets contain incomplete or unreliable fittings that are unsuitable for training high-quality 3D avatar reconstruction models. Experiments demonstrate that DynaAvatar produces visually rich and generalizable animations, outperforming prior methods. Code, pretrained models, and reannotations will be released.

---

## TODO List
- [x] Release inference code
- [] Release train code
- [ ] Release re-annotated datsets

---

## 🚀 Getting Started

### 1. Installation
To set up the environment, please refer to **[INSTALL.md](./INSTALL.md)** for step-by-step instructions.

> [!IMPORTANT]
> We performed the installation using the commands documented in `INSTALL.md`. Please ensure your **Torch** and **CUDA** versions are installed carefully to match your specific hardware environment.

### 2. Model & Data Preparation

Follow these steps to download the required models and configure the project paths.

#### **A. LHM Prior Model**
Download the Large Human Model (LHM) prior weights and extract them:
```bash
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar 
tar -xvf LHM_prior_model.tar
```

#### **B. LHM Model Configuration**
First, download the LHM model using the following script:
```bash
python download.py
```
Next, you must manually update the configuration file located at:  
`/data1/qw00n/DynaAvatar_RELEASE/pretrained_models/Damo_XR_Lab/LHM-500M/step_060000/config.json`

Add or modify these specific fields in the JSON file:
```json
{
    "n_history_length": 15,
    "is_dynamic": true
}
```

#### **C. DynaAvatar Checkpoints**
Download the DynaAvatar checkpoint and update the YAML configuration:
* **Config Path**: `/data1/qw00n/DynaAvatar_RELEASE/configs/inference/human-lrm-500M.yaml`
* **Action**: Update the `saver.load_model` field to point to the absolute path of your downloaded checkpoint.

#### **D. Additional Assets**
* **Example Motion Sequences**: Download the sample motion sequences required for inference.
* **Voxel Grid**: Ensure the voxel grid files are placed in:  
  `/data1/qw00n/DynaAvatar_RELEASE/pretrained_models/volume_voxel_grid`

---

## 💻 Inference 
Once the setup is complete, you can run the inference pipeline using the following command:

```bash
python inference.py --configs ./configs/inference/human-lrm-500M.yaml
```

---

## 🤝 Acknowledgement
This work is built upon several amazing open-source projects: [4DGS](https://github.com/hustvl/4DGaussians), [LHM](https://github.com/modelscope/LHM), and others. We are grateful for their excellent contributions to the community.

---

## 🔗 Citation
If you find our work helpful, please cite:
```bibtex
@InProceedings{Kwon_2025_CVPR,
    author    = {Kwon, Joohyun and Cho, Hanbyel and Kim, Junmo},
    title     = {DynaAvatar: Dynamic Human Avatar Synthesis with LHM Priors},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025}
}
```