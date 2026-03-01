# Copyright (c) 2023-2024, Qi Zuo & Lingteng Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# memory save version


import base64
import os
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image

torch._dynamo.config.disable = True
import argparse
import os
import pdb
import shutil
import subprocess

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from LHM.utils.model_download_utils import AutoModelQuery
from LHM.utils.model_query_utils import AutoModelSwitcher

# Initialize remove function to None initially
remove = None
try:
    from engine.SegmentAPI.SAM import SAM2Seg
    print("SAM2Seg found and imported.")
except ImportError:
    print("\033[31mSAM2Seg import failed. Attempting to import rembg...\033[0m")
    try:
        from rembg import remove
        print("rembg imported successfully.")
    except ImportError:
        print("\033[31mError: Neither SAM2Seg nor rembg could be imported. Background removal will not work.\033[0m")
except Exception as e:
    print(f"\033[31mAn unexpected error occurred during import: {e}\033[0m")
    # Attempt to import rembg as a fallback even on unexpected errors during SAM2 import
    try:
        from rembg import remove
        print("rembg imported successfully as fallback.")
    except ImportError:
         print("\033[31mError: rembg could not be imported as fallback either.\033[0m")


from engine.pose_estimation.video2motion import Video2MotionPipeline
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    resize_image_keepaspect_np,
)
from LHM.utils.download_utils import download_extract_tar_from_url, download_from_url
from LHM.utils.face_detector import VGGHeadDetector
from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.gpu_utils import check_single_gpu_memory
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.model_card import MEMORY_MODEL_CARD, MODEL_CARD, MODEL_CONFIG
from LHM.utils.video_utils import get_video_hash


def download_geo_files():
    if not os.path.exists('./pretrained_models/dense_sample_points/1_20000.ply'):
        download_from_url('https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/1_20000.ply','./pretrained_models/dense_sample_points/')

def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device

def query_model_config(model_name):
    try:
        model_params = model_name.split('-')[1]
        
        return MODEL_CONFIG[model_params] 
    except:
        return None

def prior_check():
    if not os.path.exists('./pretrained_models'):
        prior_data = MODEL_CARD['prior_model']
        download_extract_tar_from_url(prior_data)

def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    # obtain bbox
    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)

    # scale box to 1.05
    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box


def infer_preprocess_image(
    rgb_path,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]


    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard


    target_w = int(min(w * scale_ratio, h))
    if target_w - w >0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        offset_h = int(target_h - h)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0  # normalize to [0, 1]
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # resize to specific size require by preprocessor of smplx-estimator.
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    # resize to render_tgt_size for training

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:

        # ******************** Merge *********************** #
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert (
            abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
        ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert (
            abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
        ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"

        # ******************** Merge *********************** #
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 1, H, W]
    return rgb, mask, intr

def parse_configs():

    cli_cfg = OmegaConf.create()
    cfg = OmegaConf.create()

    query_model = AutoModelQuery()

    # parse from ENV
    if os.environ.get("APP_MODEL_NAME") is not None:
        model_path = query_model.query(os.environ.get("APP_MODEL_NAME"))
        model_name = os.environ.get("APP_MODEL_NAME")
    else:
        raise NotImplementedError

    cli_cfg.model_name = model_path 

    model_config = query_model_config(model_name)

    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train

def _build_model(cfg):
    from LHM.models import model_dict

    hf_model_cls = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"])
    model = hf_model_cls.from_pretrained(cfg.model_name)
    model = model.half()
    return model

def animation_infer(renderer, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors):
    '''Inference code avoid repeat forward.
    '''
    render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
        render_intrs[0, 0, 0, 2] * 2
    )
    # render target views
    render_res_list = []
    num_views = render_c2ws.shape[1]
    start_time = time.time()

    # render target views
    render_res_list = []

    for view_idx in range(num_views):
        render_res = renderer.forward_animate_gs(
            gs_model_list,
            query_points,
            renderer.get_single_view_smpl_data(smplx_params, view_idx),
            render_c2ws[:, view_idx : view_idx + 1],
            render_intrs[:, view_idx : view_idx + 1],
            render_h,
            render_w,
            render_bg_colors[:, view_idx : view_idx + 1],
        )
        render_res_list.append(render_res)
    print(
        f"time elpased(animate gs model per frame):{(time.time() -  start_time)/num_views}"
    )

    out = defaultdict(list)
    for res in render_res_list:
        for k, v in res.items():
            if isinstance(v[0], torch.Tensor):
                out[k].append(v.detach().cpu())
            else:
                out[k].append(v)
    for k, v in out.items():
        # print(f"out key:{k}")
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.concat(v, dim=1)
            if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                out[k] = out[k][0].permute(
                    0, 2, 3, 1
                )  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]
        else:
            out[k] = v
    return out

def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")

def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir

def init_preprocessor():
    from LHM.utils.preprocess import Preprocessor
    global preprocessor
    preprocessor = Preprocessor()

def preprocess_fn(image_in: np.ndarray, remove_bg: bool, recenter: bool, working_dir):
    image_raw = os.path.join(working_dir.name, "raw.png")
    with Image.fromarray(image_in) as img:
        img.save(image_raw)
    image_out = os.path.join(working_dir.name, "rembg.png")
    success = preprocessor.preprocess(image_path=image_raw, save_path=image_out, rmbg=remove_bg, recenter=recenter)
    assert success, f"Failed under preprocess_fn!"
    return image_out

def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"


@torch.no_grad()
# Removed pose_estimator, face_detector, parsing_net from function signature
def demo_lhm(lhm, motion_generation, cfg): # Only lhm and cfg remain from original models

    motion_processing_dir = './train_data/users/motion_processing'
    if os.path.exists(motion_processing_dir):
        shutil.rmtree(motion_processing_dir)
    os.makedirs(motion_processing_dir, exist_ok=True)


    @spaces.GPU(duration=100)
    def core_fn(image: str, video_params, working_dir):
        image_raw = os.path.join(working_dir.name, "raw.png")
        with Image.fromarray(image) as img:
            img.save(image_raw)
        
        
        base_vid = os.path.basename(video_params).split(".")[0]
        smplx_params_dir = os.path.join("./train_data/motion_video/", base_vid, "smplx_params")

        if not os.path.exists(smplx_params_dir):
            # user-defined motion video

            motion_processing_dir = './train_data/users/motion_processing'
            video_hash = get_video_hash(video_params)
            output_path =  os.path.join(motion_processing_dir, video_hash)

            if not os.path.exists(output_path):
                # Load motion_generation only when needed
                print("Loading Video2MotionPipeline...")
                device = avaliable_device() # Ensure device is available in this scope
                motion_generation_ondemand = Video2MotionPipeline(
                    './pretrained_models/human_model_files',
                    fitting_steps=[30, 50],
                    device=device,
                    kp_mode='vitpose',
                    visualize=False,
                    pad_ratio=0.2,
                    fov=60,
                )
                print("Generating motion parameters...")
                smplx_params_dir = motion_generation_ondemand(video_params, output_path, is_file_only=True)
                # Unload motion_generation after use
                print("Unloading Video2MotionPipeline...")
                del motion_generation_ondemand
                torch.cuda.empty_cache()
                print("Video2MotionPipeline unloaded.")
            else:
                smplx_params_dir = os.path.join(output_path, 'smplx_params')

        dump_video_path = os.path.join(working_dir.name, "output.mp4")
        dump_image_path = os.path.join(working_dir.name, "output.png")

        # prepare dump paths
        omit_prefix = os.path.dirname(image_raw)
        image_name = os.path.basename(image_raw)
        uid = image_name.split(".")[0]
        subdir_path = os.path.dirname(image_raw).replace(omit_prefix, "")
        subdir_path = (
            subdir_path[1:] if subdir_path.startswith("/") else subdir_path
        )
        print("subdir_path and uid:", subdir_path, uid)

        motion_seqs_dir = smplx_params_dir
        
        motion_name = os.path.dirname(
            motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
        )

        motion_name = os.path.basename(motion_name)

        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)

        print(image_raw, motion_seqs_dir, dump_image_dir, dump_video_path)

        dump_tmp_dir = dump_image_dir


        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30

        aspect_standard = 5.0 / 3
        motion_img_need_mask = cfg.get("motion_img_need_mask", False)  # False
        vis_motion = cfg.get("vis_motion", False)  # False

        # --- On-demand SAM2/rembg loading and execution ---
        parsing_mask = None
        parsingnet_ondemand = None
        try:
            # Try to load SAM2 on demand
            print("Attempting to load SAM2Seg...")
            parsingnet_ondemand = SAM2Seg()
            print("SAM2Seg loaded successfully.")
            with torch.no_grad():
                 parsing_out = parsingnet_ondemand(img_path=image_raw, bbox=None)
                 parsing_mask = (parsing_out.masks * 255).astype(np.uint8)
            print("Image parsed using SAM2.")
        except NameError: # Catches if SAM2Seg is not defined (import failed)
             print("SAM2Seg not available (NameError). Falling back to rembg.")
        except Exception as e: # Catches other potential errors during SAM2 loading/running
             print(f"Error during SAM2 processing: {e}. Falling back to rembg.")

        if parsing_mask is None: # If SAM2 failed or wasn't available
             try:
                 print("Using rembg for background removal...")
                 img_np = cv2.imread(image_raw)
                 remove_np = remove(img_np)
                 parsing_mask = remove_np[...,3]
                 print("Background removed using rembg.")
             except Exception as e:
                 print(f"Error during rembg processing: {e}. Cannot generate mask.")
                 # Handle error appropriately, maybe raise or return an error message
                 raise gr.Error("Failed to process image background removal.")

        # Unload SAM2 if it was loaded
        if parsingnet_ondemand is not None:
             print("Unloading SAM2Seg...")
             del parsingnet_ondemand
             torch.cuda.empty_cache()
             print("SAM2Seg unloaded.")
        # --- End of On-demand SAM2/rembg ---

        # Ensure parsing_mask is valid before proceeding
        if parsing_mask is None:
             raise gr.Error("Failed to generate parsing mask for the image.")

        # --- On-demand PoseEstimator loading and execution ---
        print("Loading PoseEstimator...")
        device = avaliable_device() # Ensure device is available
        pose_estimator_ondemand = PoseEstimator(
            "./pretrained_models/human_model_files/", device='cpu' # Load to CPU first potentially? Or directly to device? Let's use device.
        )
        pose_estimator_ondemand.to(device)
        pose_estimator_ondemand.device = device
        print("PoseEstimator loaded.")
        with torch.no_grad():
             shape_pose = pose_estimator_ondemand(image_raw)
        print("Pose estimated.")
        # Unload PoseEstimator
        shape_param_beta = shape_pose.beta # Store the result before deleting
        is_full_body = shape_pose.is_full_body # Store result
        msg = shape_pose.msg # Store result
        print("Unloading PoseEstimator...")
        del pose_estimator_ondemand
        del shape_pose # Delete intermediate variable too
        torch.cuda.empty_cache()
        print("PoseEstimator unloaded.")
        # --- End of On-demand PoseEstimator ---

        assert is_full_body, f"The input image is illegal, {msg}"

        # prepare reference image
        image, _, _ = infer_preprocess_image(
            image_raw,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )

        # --- On-demand VGGHeadDetector loading and execution ---
        src_head_rgb = None
        try:
            print("Loading VGGHeadDetector...")
            device = avaliable_device() # Ensure device is available
            facedetector_ondemand = VGGHeadDetector(
                "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
                device=device,
            )
            # facedetector_ondemand.to(device) # Already done in constructor if device is cuda
            print("VGGHeadDetector loaded.")

            rgb_face = np.array(Image.open(image_raw))[...,:3]  # RGBA input
            rgb_face = torch.from_numpy(rgb_face).permute(2, 0, 1).to(device) # Move tensor to device
            bbox = facedetector_ondemand.detect_face(rgb_face)
            head_rgb = rgb_face[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            head_rgb = head_rgb.permute(1, 2, 0)
            src_head_rgb = head_rgb.cpu().numpy() # Get result as numpy array on CPU
            print("Face detected and head cropped.")

            # Unload VGGHeadDetector
            print("Unloading VGGHeadDetector...")
            del facedetector_ondemand
            del rgb_face # Delete intermediate tensor
            del head_rgb # Delete intermediate tensor
            torch.cuda.empty_cache()
            print("VGGHeadDetector unloaded.")

        except Exception as e:
            print(f"Face detection failed or w/o head input: {e}")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)
            # Ensure unloading even if detection fails but loading succeeded
            if 'facedetector_ondemand' in locals():
                 print("Unloading VGGHeadDetector after error...")
                 del facedetector_ondemand
                 torch.cuda.empty_cache()
                 print("VGGHeadDetector unloaded.")
        # --- End of On-demand VGGHeadDetector ---


        # resize to dino size (src_head_rgb is now defined)
        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(cfg.src_head_size, cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (cfg.src_head_size, cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        save_ref_img_path = os.path.join(
            dump_tmp_dir, "output.png"
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # read motion seq
        motion_name = os.path.dirname(
            motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
        )
        motion_name = os.path.basename(motion_name)

        motion_seq = prepare_motion_seqs(
            motion_seqs_dir,
            None,
            save_root=dump_tmp_dir,
            fps=30,
            bg_color=1.0,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
            vis_motion=vis_motion,
            motion_size=3000,
        )

        camera_size = len(motion_seq["motion_seqs"])
        # shape_param = shape_pose.beta # Use the stored value
        shape_param = shape_param_beta

        device = avaliable_device() # Use the helper function
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        lhm.to(dtype)

        smplx_params = motion_seq['smplx_params']
        smplx_params['betas'] = shape_param.to(device)

        gs_model_list, query_points, transform_mat_neutral_pose = lhm.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )

        # rendering !!!!
        start_time = time.time()

        batch_list = [] 

        batch_size = 5  # avoid memeory out!

        for batch_i in range(0, camera_size, batch_size):
            with torch.no_grad():
                # TODO check device and dtype
                # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])

                print(f"batch: {batch_i}, total: {camera_size //batch_size +1} ")

                keys = [
                    "root_pose",
                    "body_pose",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                    "lhand_pose",
                    "rhand_pose",
                    "trans",
                    "focal",
                    "princpt",
                    "img_size_wh",
                    "expr",
                ]


                batch_smplx_params = dict()
                batch_smplx_params["betas"] = shape_param.to(device)
                batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
                for key in keys:
                    batch_smplx_params[key] = motion_seq["smplx_params"][key][
                        :, batch_i : batch_i + batch_size
                    ].to(device)

                # def animation_infer(self, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors, render_h, render_w):
                res = lhm.animation_infer(gs_model_list, query_points, batch_smplx_params,
                    render_c2ws=motion_seq["render_c2ws"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_intrs=motion_seq["render_intrs"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    )

            comp_rgb = res["comp_rgb"] # [Nv, H, W, 3], 0-1
            comp_mask = res["comp_mask"] # [Nv, H, W, 3], 0-1
            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
            batch_list.append(batch_rgb)

            del res
            torch.cuda.empty_cache()
        
        rgb = np.concatenate(batch_list, axis=0)
        print(f"time elapsed: {time.time() - start_time}")

        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)

            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]))[
                    None, :, :, :
                ],
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate(
                [rgb, motion_seq["vis_motion_render"], vis_ref_img], axis=2
            )

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        images_to_video(
            rgb,
            output_path=dump_video_path,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )


        return dump_image_path, dump_video_path

    _TITLE = '''LHM: Large Animatable Human Model'''

    _DESCRIPTION = '''
        <strong>Reconstruct a human avatar in 0.2 seconds with A100!</strong>
    '''

    with gr.Blocks(analytics_enabled=False) as demo:

        logo_url = "./assets/LHM_logo_parsing.png"
        logo_base64 = get_image_base64(logo_url)
        gr.HTML(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h1> <img src="{logo_base64}" style='height:35px; display:inline-block;'/> Large Animatable Human Model </h1>
            </div>
            </div>
            """
        )

        gr.Markdown(
                """
                <p align="center">
                <a title="Website" href="https://lingtengqiu.github.io/LHM/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                    <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
                </a>
                <a title="arXiv" href="https://arxiv.org/pdf/2503.10625" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                    <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
                </a>
                <a title="Github" href="https://github.com/aigc3d/LHM" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                    <img src="https://img.shields.io/github/stars/aigc3d/LHM?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
                </a>
                <a title="Video" href="https://www.youtube.com/watch?v=tivEpz_yiEo" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                    <img src="https://img.shields.io/badge/YouTube-QiuLingteng-red?logo=youtube" alt="Video">
                </a>
            """
            )

        gr.HTML(
            """<p><h4 style="color: red;"> Notes: Please input human image. Currently, it only supports motion video input with a maximum of 500 frames. </h4></p>"""
            """<p><h4 style="color: red;"> For LHM-500M, we require at least 24 GB of GPU memory; for LHM-1B, we require at least 32 GB of memory.</h4></p>"""
        )

        # DISPLAY
        with gr.Row():

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_input_image"):
                    with gr.TabItem('Input Image'):
                        with gr.Row():
                            input_image = gr.Image(label="Input Image", value="./train_data/example_imgs/00000000_joker_2.jpg",image_mode="RGBA", height=480, width=270, sources="upload", type="numpy", elem_id="content_image")
                # EXAMPLES
                examples = os.listdir('./train_data/example_imgs/')
                with gr.Row():
                    examples = [os.path.join('./train_data/example_imgs/', example) for example in examples]
                    gr.Examples(
                        examples=examples,
                        inputs=[input_image], 
                        examples_per_page=9,
                    )

            examples_video =  os.listdir('./train_data/motion_video/')
            examples =[os.path.join('./train_data/motion_video/', example, 'samurai_visualize.mp4') for example in examples_video] 

            examples = sorted(examples)
            new_examples = []
            for example in examples:
                video_basename = os.path.basename(os.path.dirname(example))
                input_video = os.path.join(os.path.dirname(example), video_basename+'.mp4')
                if not os.path.exists(input_video):
                    shutil.copyfile(example, input_video)
                new_examples.append(input_video)

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_input_video"):
                    with gr.TabItem('Target Motion'):
                        with gr.Row():
                            video_input = gr.Video(label="Input Video", sources='upload', height=480, width=270, interactive=True, value=new_examples[3])
                with gr.Row():
                    gr.Examples(
                        examples=new_examples,
                        inputs=[video_input],
                        examples_per_page=9,
                    )

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_processed_image"):
                    with gr.TabItem('Processed Image'):
                        with gr.Row():
                            processed_image = gr.Image(label="Processed Image", image_mode="RGBA", type="filepath", elem_id="processed_image", height=480, width=270, interactive=False)

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_render_video"):
                    with gr.TabItem('Rendered Video'):
                        with gr.Row():
                            output_video = gr.Video(label="Rendered Video", format="mp4", height=480, width=270, autoplay=True)

        # SETTING
        with gr.Row():
            with gr.Column(variant='panel', scale=1):
                submit = gr.Button('Generate', elem_id="openlrm_generate", variant='primary')


        working_dir = gr.State()
        submit.click(
            fn=assert_input_image,
            inputs=[input_image],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=core_fn,
            inputs=[input_image, video_input, working_dir], # video_params refer to smpl dir
            outputs=[processed_image, output_video],
        )

        demo.queue()
        demo.launch(server_name="0.0.0.0")

def get_parse():
    import argparse
    parser = argparse.ArgumentParser(description='LHM-gradio: Large Animatable Human Model')
    parser.add_argument('--model_name', default='LHM-1B-HF', type=str, choices=['LHM-500M', 'LHM-1B', 'LHM-500M-HF', 'LHM-1B-HF', "LHM-MINI"], help='Model name')
    args = parser.parse_args()
    return args


def launch_gradio_app():

    args = get_parse()
    model_name = args.model_name # Directly use the model name from args


    model_switcher = AutoModelSwitcher(MEMORY_MODEL_CARD, extra_memory=-2000)
    model_name =  model_switcher.query(model_name)

    print(f"Using specified model: {model_name}") # Add a print statement to confirm

    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": model_name,  # choice from MODEL_CARD
        "APP_TYPE": "infer.human_lrm",
        "NUMBA_THREADING_LAYER": 'omp',
    })

    prior_check()


    download_geo_files()

    device= avaliable_device()


    accelerator = Accelerator()

    cfg, cfg_train = parse_configs()
    lhm = _build_model(cfg)
    lhm.to('cuda')

    demo_lhm(lhm, None, cfg) # Removed pose_estimator and facedetector arguments




if __name__ == '__main__':
    launch_gradio_app()

