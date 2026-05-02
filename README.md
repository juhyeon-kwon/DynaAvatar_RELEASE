<div align="center">


# DynaAvatar: Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image <br> (CVPR 2026)

[**[Project Page]**](https://juhyeon-kwon.github.io/DynaAvatar.github.io/)  |  [**[Paper]**](https://arxiv.org/pdf/2603.14772)  |  [**[arXiv]**](https://arxiv.org/abs/2603.14772)  |  [**[Video]**](https://www.youtube.com/watch?v=50e8RyGcxwc)
</div>

---

## TODO
- [x] Release inference code 💃
- [x] Release training code 🕺
- [x] Release demo using in-the-wild video' motion 
- [ ] Release re-annotated datasets 📁

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
Here are three examples using different motion sequences (e.g., DNA-Rendering, 4D-DRESS datasets, and in-the-wild sequence). 

#### **Example 1: DNA-Rendering sequence**
```bash
CUDA_VISIBLE_DEVICES=0 bash inference.sh LHM-500M \
    ./assets/novel_subject \
    PATH/motion_seqs/DNA_Rendering/0124_03/smplx/smplx_params_smooth \
    None 0 500 15
```

#### **Example 2: 4D-DRESS sequence**
```bash
CUDA_VISIBLE_DEVICES=0 bash inference.sh LHM-500M \
    ./assets/novel_subject \
    PATH/motion_seqs/4D-DRESS/00152_outer_16/smplx/smplx_params_smooth \
    None 0 500 30
```

#### **Example 3: In-the-wild sequence**
Download model weights for motion prediction from the given video.
```bash
wget -P ./pretrained_models/human_model_files/pose_estimate https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/yolov8x.pt
wget -P ./pretrained_models/human_model_files/pose_estimate https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/vitpose-h-wholebody.pth
```
Install dependencies.
```bash
cd ./engine/pose_estimation
pip install mmcv==1.3.9 --no-build-isolation
pip install -v -e third-party/ViTPose
pip install ultralytics
```
Run the motion prediction script.
```bash
python ./engine/pose_estimation/video2motion.py --video_path ${VIDEO_PATH} 
```
After motion prediction, the predicted SMPL-X parameters and camera parameters will be saved in `./custom_motion/<video_name_without_extension>`.  
Finally, run the inference script: 
```bash
CUDA_VISIBLE_DEVICES=0 bash inference.sh LHM-500M \
    ${REF_IMG_PATH} \
    custom_motion/<video_name_without_extension>/smplx/smplx_params_smooth \
    None 0 None ${VIDEO_FPS}
```

### Arguments
```bash
bash inference.sh [MODEL_NAME] [SOURCE_IMAGE_DIR] [MOTION_PARAM_PATH] [BG_PATH] [START_FRAME] [MOTION_SIZE] [VIDEO_FPS]
```

* **MODEL_NAME**: Name of the backbone model (e.g., `LHM-500M`).
* **SOURCE_IMAGE_DIR**: Path to the folder containing source images.
* **MOTION_PARAM_PATH**: Path to the SMPL-X motion parameters.
* **MOTION_SIZE**: Maximum number of frames to render (e.g., `300`).
* **FPS**: Frames per second (e.g., `15` or `30`).

> 💡 **Note:** If you encounter VRAM issues, please adjust the `batch_size` in [`DynaAvatar_RELEASE/LHM/runners/infer/human_lrm.py`](https://github.com/juhyeon-kwon/DynaAvatar_RELEASE/blob/4f5f372a987deb3aa8d8f2ee2fb900361b1df9dd/LHM/runners/infer/human_lrm.py#L969C1-L970C1) at line **969**.
---

## 📁 Reannotated Datasets

First, download original datasets from [4D-DRESS](https://github.com/eth-ait/4d-dress) and [DNA-Rendering](https://github.com/DNA-Rendering/DNA-Rendering).
You can download our reannotated datasets [here](https://drive.google.com/drive/folders/1QMLCWHvimh3ZZ6g2EHvCuc28SmGNCZUi?usp=drive_link).

> 💡 **Note:** The shared SMPL-X parameters are re-annotated results. These are numerical values that **cannot be used to reconstruct the original dataset without access to the raw images from 4D-DRESS and DNA-Rendering**. We provide these for research reproducibility only.
### 1. 4D-DRESS
#### 1.1 Render Images and Save Camera Parameters
```bash
python ./preprocess/render_4d_dress.py --dataset_root_dir /PATH/TO/4D-DRESS --target_root_dir /PATH/TO/4D-DRESS_reannot_release
```
#### 1.2 Archive & Make TOC (for Training)
Due to our file system limitations on our server, we store the dataset in .tar format.
```bash
python ./preprocess/archive_as_tar.py --dataset_root_dir /PATH/TO/4D-DRESS_reannot_release --target_dir /PATH/TO/4D-DRESS_reannot_tar
python ./preprocess/make_toc.py --target_dir /PATH/TO/4D-DRESS_reannot_tar
```


### 2. DNA-Rendering
#### 2.1 Extract Images and Camera Parameters from the Original Dataset
```bash
pip install h5py
python ./preprocess/extract_from_smc.py --input_dir /PATH/TO/dna_rendering_release_data/Part {idx}/dna_rendering_part{idx}_main --output_root_dir /PATH/TO/DNA_Rendering
```
#### 2.2 Crop Original Images and Save Rescaled Camera Parameters
We crop the original images to remove background regions.
```bash
python ./preprocess/crop_dna_rendering.py --dataset_root_dir /PATH/TO/DNA_Rendering --target_root_dir /PATH/TO/DNA_Rendering_reannot_release
```
#### 2.3 Archive & Make TOC (for Training)
```bash
python ./preprocess/archive_as_tar.py --dataset_root_dir /PATH/TO/DNA_Rendering_reannot_release --target_dir /PATH/TO/DNA_Rendering_reannot_tar
python ./preprocess/make_toc.py --target_dir /PATH/TO/DNA_Rendering_reannot_tar
```

---

## 🕺 Training 
### 1. Configuration Setup
Before starting, you need to update the configuration files with your local environment settings:

#### 1.1 Set Data Paths
Open ```PATH/DynaAvatar_RELEASE/configs/train/human-lrm-500M-dynamic.yaml```. Use ```Ctrl+F``` to find all occurrences of ```PATH``` and replace them with your actual absolute project path.

#### 1.2 Adjust Multi-GPU Settings
In ```PATH/DynaAvatar_RELEASE/configs/accelerate-train.yaml```, modify the ```num_processes``` value according to the number of GPUs you intend to use.

#### 1.3 Select Target GPUs
Open ```PATH/DynaAvatar_RELEASE/train_dynaavatar.sh``` and set the ```CUDA_VISIBLE_DEVICES``` to the specific GPU IDs you wish to allocate for training.


### 2. Run Training
Once the configuration is complete, you can simply launch the training with the following command:

```bash
bash train_dynaavatar.sh
```
---

## Acknowledgement
This work is built upon several amazing open-source projects: [PERSONA](https://github.com/mks0601/PERSONA_RELEASE), [LHM](https://github.com/aigc3d/LHM), and others. We are grateful for their excellent contributions to the community.

---

## Citation
If you find our work helpful, please cite:
```bibtex
@inproceedings{kwon2026dynavatar,
  title={Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image},
  author={Kwon, Joohyun and Sim, Geonhee and Moon, Gyeongsik},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```