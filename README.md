# DynaAvatar: Dynamic Human Avatar Synthesis with LHM Priors

## CVPR 2025

### [[Project Page]](#) | [[Paper]](#) | [[arXiv]](#) | [[Poster]](#) | [Video](#)

**This is the official PyTorch implementation of the approach described in the following paper:**
> **DynaAvatar: Dynamic Human Avatar Synthesis with LHM Priors**\
> [Your Name*], [Co-Author*], and [Junmo Kim†]\
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025

---

## 🏠 Overview
DynaAvatar is a high-performance framework designed for high-fidelity dynamic human avatar synthesis. By integrating Large Human Model (LHM) priors with a 4D Gaussian-based representation, our method achieves superior temporal consistency and visual quality when reconstructing complex human motions.

![overall_framework](figs/pipeline.jpg)
> Overall pipeline of DynaAvatar. Our method effectively separates static and dynamic components to optimize the synthesis process for realistic human motion.

---

## 📝 TODO List
- [x] Release core inference code
- [x] Release LHM prior integration modules
- [x] Provide pre-trained checkpoints
- [ ] Release training scripts and data processing tools

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