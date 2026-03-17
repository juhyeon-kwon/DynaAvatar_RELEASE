# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-10 18:56:08
# @Function      : FUNCTION_DESCRIPTION

import glob
import json
import math
import os
import pdb
from collections import defaultdict

import cv2
import decord
import numpy as np
import torch
from PIL import Image
from pytorch3d.io import save_ply
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from os.path import exists


def generate_rotation_matrix_y(degrees):
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)


def scale_intrs(intrs, ratio_x, ratio_y):
    if len(intrs.shape) >= 3:
        intrs[:, 0] = intrs[:, 0] * ratio_x
        intrs[:, 1] = intrs[:, 1] * ratio_y
    else:
        intrs[0] = intrs[0] * ratio_x
        intrs[1] = intrs[1] * ratio_y
    return intrs


def calc_new_tgt_size(cur_hw, tgt_size, multiply):
    ratio = tgt_size / min(cur_hw)
    tgt_size = int(ratio * cur_hw[0]), int(ratio * cur_hw[1])
    tgt_size = (
        int(tgt_size[0] / multiply) * multiply,
        int(tgt_size[1] / multiply) * multiply,
    )
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x


def calc_new_tgt_size_by_aspect(cur_hw, aspect_standard, tgt_size, multiply):
    assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
    tgt_size = tgt_size * aspect_standard, tgt_size
    tgt_size = (
        int(tgt_size[0] / multiply) * multiply,
        int(tgt_size[1] / multiply) * multiply,
    )
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x


def _load_pose(pose):
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()

    c2w = torch.eye(4)
    # c2w[:3, :3] = torch.tensor(pose["R"])
    # c2w[3, :3] = torch.tensor(pose["t"])
    c2w = c2w.float()

    return c2w, intrinsic


def img_center_padding(img_np, pad_ratio):

    ori_w, ori_h = img_np.shape[:2]

    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)

    if len(img_np.shape) > 2:
        img_pad_np = np.zeros((w, h, img_np.shape[2]), dtype=np.uint8)
    else:
        img_pad_np = np.zeros((w, h), dtype=np.uint8)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[
        offset_h : offset_h + img_np.shape[0] :, offset_w : offset_w + img_np.shape[1]
    ] = img_np

    return img_pad_np


def resize_image_keepaspect_np(img, max_tgt_size):
    """
    similar to ImageOps.contain(img_pil, (img_size, img_size)) # keep the same aspect ratio
    """
    h, w = img.shape[:2]
    ratio = max_tgt_size / max(h, w)
    new_h, new_w = round(h * ratio), round(w * ratio)
    return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop_according_to_mask(img, mask, aspect_standard, enlarge_ratio):
    """
    img: [H, W, 3]
    mask: [H, W]
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise Exception("empty mask")

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    half_w = max(abs(center_x - x_min), abs(center_x - x_max))
    half_h = max(abs(center_y - y_min), abs(center_y - y_max))
    half_w_raw = half_w
    half_h_raw = half_h
    aspect = half_h / half_w

    if aspect >= aspect_standard:
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)

    # not exceed original image
    if half_h > center_y:
        half_w = round(half_h_raw / aspect_standard)
        half_h = half_h_raw
    if half_w > center_x:
        half_h = round(half_w_raw * aspect_standard)
        half_w = half_w_raw

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
        enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
        enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
        enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
        enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
        enlarge_ratio_cur = (
            np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min)
            + enlarge_ratio_min
        )
        half_h, half_w = round(enlarge_ratio_cur * half_h), round(
            enlarge_ratio_cur * half_w
        )

    assert half_h <= center_y
    assert half_w <= center_x
    assert abs(half_h / half_w - aspect_standard) < 0.03

    offset_x = center_x - half_w
    offset_y = center_y - half_h

    new_img = img[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]
    new_mask = mask[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]

    return new_img, new_mask, offset_x, offset_y


def preprocess_image(
    rgb_path,
    mask_path,
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
    if pad_ratio > 0:
        rgb = img_center_padding(rgb, pad_ratio)

    rgb = rgb / 255.0  # normalize to [0, 1]
    if need_mask:
        if rgb.shape[2] < 4:
            if mask_path is not None:
                mask = np.array(Image.open(mask_path))
            else:
                from rembg import remove

                # rembg cuda version -> error
                mask = remove(rgb_raw[:, :, (2, 1, 0)])[:, :, -1]  # np require [bgr]
                print("rmbg mask: ", mask.min(), mask.max(), mask.shape)
            if pad_ratio > 0:
                mask = img_center_padding(mask, pad_ratio)
            mask = mask / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]  # [H, W]
    else:
        # just placeholder
        mask = np.ones_like(rgb[:, :, 0])

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


def extract_imgs_from_video(video_file, save_root, fps):
    print(f"extract_imgs_from_video:{video_file}")
    vr = decord.VideoReader(video_file)
    for i in range(0, len(vr), fps):
        frame = vr[i].asnumpy()
        save_path = os.path.join(save_root, f"{i:05d}.jpg")
        cv2.imwrite(save_path, frame[:, :, (2, 1, 0)])


def predict_motion_seqs_from_images(image_folder: str, save_root, fps=6):
    id_name = os.path.splitext(os.path.basename(image_folder))[0]
    if os.path.isfile(image_folder) and (
        image_folder.endswith("mp4") or image_folder.endswith("move")
    ):
        save_frame_root = os.path.join(save_root, "extracted_frames", id_name)
        if not os.path.exists(save_frame_root):
            os.makedirs(save_frame_root, exist_ok=True)
            extract_imgs_from_video(
                video_file=image_folder, save_root=save_frame_root, fps=fps
            )
        else:
            print("skip extract_imgs_from_video......")
        image_folder = save_frame_root

    image_folder_abspath = os.path.abspath(image_folder)
    print(f"predict motion seq:{image_folder_abspath}")
    save_smplx_root = image_folder + "_smplx_params_mhmr"
    if not os.path.exists(save_smplx_root):
        cmd = f"cd thirdparty/multi-hmr &&  python infer_batch.py  --data_root {image_folder_abspath}  --out_folder {image_folder_abspath} --crop_head   --crop_hand   --pad_ratio 0.2 --smplify"
        os.system(cmd)
    else:
        print("skip predict smplx.........")
    return save_smplx_root, image_folder


def render_smplx_mesh(
    smplx_params, render_intrs, human_model_path="./pretrained_models/human_model_files"
):
    from LHM.models.rendering.smplx import smplx
    from LHM.models.rendering.smplx.vis_utils import render_mesh

    layer_arg = {
        "create_global_orient": False,
        "create_body_pose": False,
        "create_left_hand_pose": False,
        "create_right_hand_pose": False,
        "create_jaw_pose": False,
        "create_leye_pose": False,
        "create_reye_pose": False,
        "create_betas": False,
        "create_expression": False,
        "create_transl": False,
    }

    smplx_layer = smplx.create(
        human_model_path,
        "smplx",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=100,
        use_pca=False,
        use_face_contour=False,
        flat_hand_mean=True,
        **layer_arg,
    )

    body_pose = smplx_params["body_pose"]
    num_view = body_pose.shape[0]
    shape_param = smplx_params["betas"]
    if "expr" not in smplx_params:
        # supports v2.0 data format
        smplx_params["expr"] = torch.zeros((num_view, 100))

    output = smplx_layer(
        global_orient=smplx_params["root_pose"],
        body_pose=smplx_params["body_pose"].view(num_view, -1),
        left_hand_pose=smplx_params["lhand_pose"].view(num_view, -1),
        right_hand_pose=smplx_params["rhand_pose"].view(num_view, -1),
        jaw_pose=smplx_params["jaw_pose"],
        leye_pose=smplx_params["leye_pose"],
        reye_pose=smplx_params["reye_pose"],
        expression=smplx_params["expr"],
        betas=smplx_params["betas"].unsqueeze(0).repeat(num_view, 1),  # 10 blendshape
        transl=smplx_params["trans"],
        face_offset=None,
        joint_offset=None,
    )

    smplx_face = smplx_layer.faces.astype(np.int64)
    mesh_render_list = []
    for v_idx in range(num_view):
        intr = render_intrs[v_idx]
        cam_param = {
            "focal": torch.tensor([intr[0, 0], intr[1, 1]]),
            "princpt": torch.tensor([intr[0, 2], intr[1, 2]]),
        }
        render_shape = int(cam_param["princpt"][1] * 2), int(
            cam_param["princpt"][0] * 2
        )  # require h, w
        mesh_render, is_bkg = render_mesh(
            output.vertices[v_idx],
            smplx_face,
            cam_param,
            np.ones((render_shape[0], render_shape[1], 3), dtype=np.float32) * 255,
            return_bg_mask=True,
        )
        mesh_render = mesh_render.astype(np.uint8)
        mesh_render_list.append(mesh_render)
    mesh_render = np.stack(mesh_render_list)
    return mesh_render


def prepare_motion_seqs(
    motion_seqs_dir,
    image_folder,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
    vis_motion=False,
    motion_size=3000,  # only support 100s videos
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.
        vis_motion (bool, optional): Flag indicating whether to visualize motion. Defaults to False.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions
            'vis_motion_render': rendering smplx motion

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """

    if motion_seqs_dir is None:
        assert image_folder is not None
        motion_seqs_dir, image_folder = predict_motion_seqs_from_images(
            image_folder, save_root, fps
        )

    motion_seqs = sorted(glob.glob(os.path.join(motion_seqs_dir, "*.json")))
    motion_seqs = motion_seqs[:motion_size]

    # source images
    c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
    smplx_params = []
    shape_param = None

    for idx, smplx_path in enumerate(motion_seqs):

        if image_folder is not None:
            file_name = os.path.splitext(os.path.basename(smplx_path))[0]
            frame_path = os.path.join(image_folder, file_name + ".png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(image_folder, file_name + ".jpg")
        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }

        if idx == 0:
            shape_param = smplx_param["betas"]

        c2w, intrinsic = _load_pose(smplx_param)
        intrinsic_raw = intrinsic.clone()
        if "expr" not in smplx_raw_data:
            # supports v2.0 data format
            max_tgt_size = int(max(smplx_param["img_size_wh"]))
        else:
            max_tgt_size = int(smplx_param["img_size_wh"][0])
            smplx_param.pop("expr")

        flame_path = smplx_path.replace("smplx_params", "flame_params")
        smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        # intrs.append(intrinsic_raw)
        smplx_params.append(smplx_param)

    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
    intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
    bg_colors = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )  # [N, 3]

    if len(rgbs) > 0:
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
        # masks = torch.cat(masks, dim=0)  # [N, 1, H, W]

    smplx_params_tmp = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)  # [Nv, xx, xx]
    smplx_params = smplx_params_tmp

    # TODO check different betas for same person
    #smplx_params["betas"] = shape_param # qw00n; useless

    if vis_motion:
        motion_render = render_smplx_mesh(smplx_params, intrs)
    else:
        motion_render = None

    # add batch dim
    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
        # print(k, smplx_params[k].shape, "motion_seq")
    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)
    if len(rgbs) > 0:
        rgbs = rgbs.unsqueeze(0)
    # print(f"c2ws:{c2ws.shape}, intrs:{intrs.shape}, rgbs:{rgbs.shape if len(rgbs) > 0 else None}")

    motion_seqs_ret = {}
    motion_seqs_ret["render_c2ws"] = c2ws
    motion_seqs_ret["render_intrs"] = intrs
    motion_seqs_ret["render_bg_colors"] = bg_colors
    motion_seqs_ret["smplx_params"] = smplx_params
    motion_seqs_ret["rgbs"] = rgbs
    motion_seqs_ret["vis_motion_render"] = motion_render
    motion_seqs_ret["motion_seqs"] = motion_seqs

    return motion_seqs_ret


def prepare_motion_seqs_2(
    motion_seqs_dir,
    image_folder,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
    vis_motion=False,
    motion_st_idx=0,
    motion_size=3000,  # only support 100s videos

    cam_param_path=None,
    camera_name='26', 
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.
        vis_motion (bool, optional): Flag indicating whether to visualize motion. Defaults to False.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions
            'vis_motion_render': rendering smplx motion

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """
    intrinsic, c2w = load_camera_params(cam_param_path, camera_name)
    if motion_seqs_dir is None:
        assert image_folder is not None
        motion_seqs_dir, image_folder = predict_motion_seqs_from_images(
            image_folder, save_root, fps
        )

    motion_seqs = glob.glob(os.path.join(motion_seqs_dir, "*.json"))
    motion_seqs.sort(key=lambda x: int(x.split('/')[-1][:-5]))
    motion_seqs = motion_seqs[motion_st_idx:motion_st_idx+motion_size]

    # source images
    c2ws, intrs, rgbs, bg_colors = [], [], [], []
    smplx_params = []
    shape_param = None

    for idx, smplx_path in enumerate(motion_seqs):

        if image_folder is not None:
            file_name = os.path.splitext(os.path.basename(smplx_path))[0]
            frame_path = os.path.join(image_folder, file_name + ".png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(image_folder, file_name + ".jpg")
        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }            

        flame_path = smplx_path.replace("smplx_params", "flame_params")
        _expr = torch.zeros(100)
        if 'expr' in smplx_param: 
            _expr[:smplx_param["expr"].shape[0]] = smplx_param["expr"]
        smplx_param["expr"] = _expr

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        smplx_params.append(smplx_param)

    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
    intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
    bg_colors = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )  # [N, 3]

    if len(rgbs) > 0:
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]

    smplx_params_tmp = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)  # [Nv, xx, xx]
    smplx_params = smplx_params_tmp

    motion_render = None

    # add batch dim
    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
        # print(k, smplx_params[k].shape, "motion_seq")
    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)
    if len(rgbs) > 0:
        rgbs = rgbs.unsqueeze(0)
    # print(f"c2ws:{c2ws.shape}, intrs:{intrs.shape}, rgbs:{rgbs.shape if len(rgbs) > 0 else None}")

    motion_seqs_ret = {}
    motion_seqs_ret["render_c2ws"] = c2ws
    motion_seqs_ret["render_intrs"] = intrs
    motion_seqs_ret["render_bg_colors"] = bg_colors
    motion_seqs_ret["smplx_params"] = smplx_params
    motion_seqs_ret["rgbs"] = rgbs
    motion_seqs_ret["vis_motion_render"] = motion_render
    motion_seqs_ret["motion_seqs"] = motion_seqs

    return motion_seqs_ret


def prepare_motion_single(
    motion_seqs_dir,
    image_name,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
    vis_motion=False,
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.
        vis_motion (bool, optional): Flag indicating whether to visualize motion. Defaults to False.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions
            'vis_motion_render': rendering smplx motion

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """

    motion_seqs = [
        os.path.join(
            motion_seqs_dir,
            image_name.replace(".jpg", ".png").replace(".png", ".json"),
        )
        for _ in range(4)
    ]
    axis_list = [0, 60, 180, 240]

    # source images
    c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
    smplx_params = []
    shape_param = None

    for idx, smplx_path in enumerate(motion_seqs):

        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }

        if idx == 0:
            shape_param = smplx_param["betas"]

        c2w, intrinsic = _load_pose(smplx_param)
        intrinsic_raw = intrinsic.clone()
        if "expr" not in smplx_raw_data:
            # supports v2.0 data format
            max_tgt_size = int(max(smplx_param["img_size_wh"]))
        else:
            max_tgt_size = int(smplx_param["img_size_wh"][0])
            smplx_param.pop("expr")

        flame_path = smplx_path.replace("smplx_params", "flame_params")
        if os.path.exists(flame_path):
            with open(flame_path) as f:
                flame_param = json.load(f)
                smplx_param["expr"] = torch.FloatTensor(flame_param["expcode"])

                # replace with flame's jaw_pose
                smplx_param["jaw_pose"] = torch.FloatTensor(flame_param["posecode"][3:])
                smplx_param["leye_pose"] = torch.FloatTensor(flame_param["eyecode"][:3])
                smplx_param["reye_pose"] = torch.FloatTensor(flame_param["eyecode"][3:])

        else:
            smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        root_rotate_matrix = axis_angle_to_matrix(smplx_param["root_pose"])
        rotate = generate_rotation_matrix_y(axis_list[idx])
        rotate = torch.from_numpy(rotate).float()
        rotate = rotate @ root_rotate_matrix
        new_rotate_axis = matrix_to_axis_angle(rotate)

        smplx_param["root_pose"] = new_rotate_axis

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        # intrs.append(intrinsic_raw)
        smplx_params.append(smplx_param)

    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
    intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
    bg_colors = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )  # [N, 3]

    if len(rgbs) > 0:
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
        # masks = torch.cat(masks, dim=0)  # [N, 1, H, W]

    smplx_params_tmp = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)  # [Nv, xx, xx]
    smplx_params = smplx_params_tmp
    # TODO check different betas for same person
    smplx_params["betas"] = shape_param

    if vis_motion:
        motion_render = render_smplx_mesh(smplx_params, intrs)
    else:
        motion_render = None

    # add batch dim
    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
        # print(k, smplx_params[k].shape, "motion_seq")
    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)
    if len(rgbs) > 0:
        rgbs = rgbs.unsqueeze(0)
    # print(f"c2ws:{c2ws.shape}, intrs:{intrs.shape}, rgbs:{rgbs.shape if len(rgbs) > 0 else None}")

    motion_seqs = {}
    motion_seqs["render_c2ws"] = c2ws
    motion_seqs["render_intrs"] = intrs
    motion_seqs["render_bg_colors"] = bg_colors
    motion_seqs["smplx_params"] = smplx_params
    motion_seqs["rgbs"] = rgbs
    motion_seqs["vis_motion_render"] = motion_render

    return motion_seqs


def prepare_motion_lrmbench(
    motion_seqs_dir,
    image_name,
    save_root,
    fps,
    bg_color,
    aspect_standard,
    enlarge_ratio,
    render_image_res,
    need_mask,
    multiply=16,
    vis_motion=False,
):
    """
    Prepare motion sequences for rendering.

    Args:
        motion_seqs_dir (str): Directory path of motion sequences.
        image_folder (str): Directory path of source images.
        save_root (str): Directory path to save the motion sequences.
        fps (int): Frames per second for the motion sequences.
        bg_color (tuple): Background color in RGB format.
        aspect_standard (float): Standard human aspect ratio (height/width).
        enlarge_ratio (float): Ratio to enlarge the source images.
        render_image_res (int): Resolution of the rendered images.
        need_mask (bool): Flag indicating whether masks are needed.
        multiply (int, optional): Multiply factor for image size. Defaults to 16.
        vis_motion (bool, optional): Flag indicating whether to visualize motion. Defaults to False.

    Returns:
        dict: Dictionary containing the prepared motion sequences.
            'render_c2ws': camera to world matrix  [B F 4 4]
            'render_intrs': intrins matrix [B F 4 4]
            'render_bg_colors' bg_colors [B F 3]
            'smplx_params': smplx_params -> ['betas',
                        'root_pose', 'body_pose',
                        'jaw_pose', 'leye_pose',
                        'reye_pose', 'lhand_pose',
                        'rhand_pose',
                        'trans', 'expr', 'focal',
                        'princpt',
                        'img_size_wh'
                ]
            'rgbs': imgs w.r.t motions
            'vis_motion_render': rendering smplx motion

    Raises:
        AssertionError: If motion_seqs_dir is None and image_folder is None.

    """
    import json

    axis_list = list(range(0, 360, 30))

    motion_seqs = [
        os.path.join(
            motion_seqs_dir,
            image_name.replace(".jpg", ".png")
            .replace(".jpeg", ".png")
            .replace(".PNG", ".png")
            .replace(".png", ".json"),
        )
        for _ in range(len(axis_list))
    ]

    # source images
    c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
    smplx_params = []
    shape_param = None

    for idx, smplx_path in enumerate(motion_seqs):

        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }

        if idx == 0:
            shape_param = smplx_param["betas"]

        c2w, intrinsic = _load_pose(smplx_param)
        intrinsic_raw = intrinsic.clone()
        if "expr" not in smplx_raw_data:
            # supports v2.0 data format
            max_tgt_size = int(max(smplx_param["img_size_wh"]))
        else:
            max_tgt_size = int(smplx_param["img_size_wh"][0])
            smplx_param.pop("expr")

        flame_path = smplx_path.replace("smplx_params", "flame_params")
        if os.path.exists(flame_path):
            with open(flame_path) as f:
                flame_param = json.load(f)
                smplx_param["expr"] = torch.FloatTensor(flame_param["expcode"])

                # replace with flame's jaw_pose
                smplx_param["jaw_pose"] = torch.FloatTensor(flame_param["posecode"][3:])
                smplx_param["leye_pose"] = torch.FloatTensor(flame_param["eyecode"][:3])
                smplx_param["reye_pose"] = torch.FloatTensor(flame_param["eyecode"][3:])

        else:
            smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        smplx_param["expr"] = torch.FloatTensor([0.0] * 100)

        root_rotate_matrix = axis_angle_to_matrix(smplx_param["root_pose"])
        rotate = generate_rotation_matrix_y(axis_list[idx])
        rotate = torch.from_numpy(rotate).float()
        rotate = rotate @ root_rotate_matrix
        new_rotate_axis = matrix_to_axis_angle(rotate)

        smplx_param["root_pose"] = new_rotate_axis

        c2ws.append(c2w)
        bg_colors.append(bg_color)
        intrs.append(intrinsic)
        # intrs.append(intrinsic_raw)
        smplx_params.append(smplx_param)

    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
    intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
    bg_colors = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )  # [N, 3]

    if len(rgbs) > 0:
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
        # masks = torch.cat(masks, dim=0)  # [N, 1, H, W]

    smplx_params_tmp = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)  # [Nv, xx, xx]
    smplx_params = smplx_params_tmp
    # TODO check different betas for same person
    smplx_params["betas"] = shape_param

    if vis_motion:
        motion_render = render_smplx_mesh(smplx_params, intrs)
    else:
        motion_render = None

    # add batch dim
    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
        # print(k, smplx_params[k].shape, "motion_seq")
    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)
    if len(rgbs) > 0:
        rgbs = rgbs.unsqueeze(0)

    motion_seqs = {}
    motion_seqs["render_c2ws"] = c2ws
    motion_seqs["render_intrs"] = intrs
    motion_seqs["render_bg_colors"] = bg_colors
    motion_seqs["smplx_params"] = smplx_params
    motion_seqs["rgbs"] = rgbs
    motion_seqs["vis_motion_render"] = motion_render

    return motion_seqs


# vcai
def load_camera_params(cam_param_path: str, camera_name: str) -> tuple[torch.Tensor, torch.Tensor]: 
    with open(cam_param_path) as f: 
        cam_params_dict = json.load(f)
    cam_param = {k: torch.tensor(v, dtype=torch.float32) for k, v in cam_params_dict[camera_name].items()}
    # intrinsic
    K_3x3 = torch.zeros((3,3), dtype=torch.float32)
    K_3x3[0,0], K_3x3[1,1] = cam_param['focal'][0], cam_param['focal'][1]
    K_3x3[0,2], K_3x3[1,2] = cam_param['princpt'][0], cam_param['princpt'][1]
    K_4x4 = torch.eye(4, dtype=torch.float32)
    K_4x4[:3, :3] = K_3x3
    
    # extrinsic: w2c -> c2w 
    R_3x3 = cam_param['R']
    Tvec = cam_param['t'].view(3,1)
    R_c2w = R_3x3.T
    T_c2w = -R_3x3.T @ Tvec
    c2w_4x4 = torch.eye(4, dtype=torch.float32)
    c2w_4x4[:3, :3] = R_c2w
    c2w_4x4[:3, 3] = T_c2w.view(3)
    
    return K_4x4, c2w_4x4
    
    
def load_camera_params_from_yml(intri_path: str, extri_path: str, camera_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OpenCV .yml 파일 형식으로 저장된 카메라 내부/외부 파라미터를 로드합니다.

    Args:
        intri_path (str): 내부 파라미터 파일(.yml) 경로.
        extri_path (str): 외부 파라미터 파일(.yml) 경로.
        camera_name (str): 파일 내에서 파라미터를 찾을 때 사용할 카메라 이름 (예: '230724').

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - intrinsic_matrix (torch.Tensor): [4, 4] 모양의 내부 파라미터 텐서.
            - c2w_matrix (torch.Tensor): [4, 4] 모양의 카메라-to-월드 변환 행렬 텐서.
    """
    assert exists(intri_path), f"Intrinsics file not found: {intri_path}"
    assert exists(extri_path), f"Extrinsics file not found: {extri_path}"
    
    fs_intri = None
    fs_extri = None
    try:
        fs_intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
        fs_extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)

        # --- Intrinsic Matrix ---
        k_key = f'K_{camera_name}'
        K_3x3 = fs_intri.getNode(k_key).mat()
        if K_3x3 is None:
            raise ValueError(f"'{k_key}' not found in the intrinsics file.")
        
        K_4x4 = np.eye(4, dtype=np.float32)
        K_4x4[:3, :3] = K_3x3.astype(np.float32)
        
        # --- Extrinsic Matrix (World-to-Camera) ---
        Rvec = fs_extri.getNode(f'R_{camera_name}').mat()
        Tvec = fs_extri.getNode(f'T_{camera_name}').mat()

        if Rvec is not None:
            R_3x3, _ = cv2.Rodrigues(Rvec)
        else:
            R_3x3 = fs_extri.getNode(f'Rot_{camera_name}').mat()

        if R_3x3 is None or Tvec is None:
            raise ValueError(f"Rotation/Translation for '{camera_name}' not found.")
        
        # World-to-Camera(w2c) -> Camera-to-World(c2w)
        R_c2w = R_3x3.T
        T_c2w = -R_3x3.T @ Tvec
        
        c2w_4x4 = np.eye(4, dtype=np.float32)
        c2w_4x4[:3, :3] = R_c2w
        c2w_4x4[:3, 3] = T_c2w.flatten()
        
        intrinsic_matrix = torch.from_numpy(K_4x4)
        c2w_matrix = torch.from_numpy(c2w_4x4)
            
        return intrinsic_matrix, c2w_matrix

    finally:
        if fs_intri is not None: fs_intri.release()
        if fs_extri is not None: fs_extri.release()

# vcai
def prepare_motion_seqs_from_npz(
    motion_npz_dir: str,
    intri_path: str,
    extri_path: str,
    camera_name: str,
    bg_color: float = 1.0,
    motion_size: int = 3000,
    num_betas= 10,
    num_exprs=100
):
    """
    'full_pose'와 'transl' 키를 포함하는 .npz 파일들을 파싱하여 렌더링에 필요한 모션 시퀀스를 준비합니다.
    'full_pose'는 root_pose, body_pose 등으로 분할됩니다.

    Args:
        motion_npz_dir (str): SMPL-X .npz 파일들이 포함된 디렉토리 경로.
        intri_path (str): 카메라 내부 파라미터 파일(.yml) 경로.
        extri_path (str): 카메라 외부 파라미터 파일(.yml) 경로.
        camera_name (str): 파라미터 파일 내에서 사용할 카메라 이름.
        bg_color (float, optional): 배경색. Defaults to 1.0.
        motion_size (int, optional): 사용할 최대 프레임 수. Defaults to 3000.

    Returns:
        dict: 원본 함수와 동일한 구조를 가진, 준비된 모션 시퀀스 딕셔너리.
    """
    intrinsic, c2w = load_camera_params(intri_path, extri_path, camera_name)

    motion_files = sorted(glob.glob(os.path.join(motion_npz_dir, "*.npz")))
    motion_files = motion_files[:motion_size]
    
    if not motion_files:
        raise FileNotFoundError(f"No .npz files found in {motion_npz_dir}")

    smplx_params_list = []
    c2ws, intrs, bg_colors = [], [], []
    
    for npz_path in motion_files:
        data = np.load(npz_path)
        full_pose = torch.from_numpy(data['fullpose']).float() 

        # 'full_pose'를 SMPL-X 각 파라미터에 맞게 슬라이싱
        smplx_param = {
            'root_pose':  full_pose[0:1],         
            'body_pose':  full_pose[1:22],       
            'jaw_pose':   full_pose[22:23],       
            'leye_pose':  full_pose[23:24],       
            'reye_pose':  full_pose[24:25],       
            'lhand_pose': full_pose[25:40],     
            'rhand_pose': full_pose[40:55],

            'betas': torch.from_numpy(np.squeeze(data['betas'])[:num_betas]),
            'expr': torch.zeros(1, num_exprs), 
            'trans': torch.from_numpy(data['transl'])
        }

        smplx_params_list.append(smplx_param)
        c2ws.append(c2w)
        intrs.append(intrinsic)
        bg_colors.append(bg_color)

    c2ws = torch.stack(c2ws, dim=0)    # [F, 4, 4]
    intrs = torch.stack(intrs, dim=0)  # [F, 4, 4]
    bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3) # [F, 3]

    smplx_params_stacked = defaultdict(list)
    for smplx in smplx_params_list:
        for k, v in smplx.items():
            smplx_params_stacked[k].append(v.squeeze(0))
            
    for k, v in smplx_params_stacked.items():
        smplx_params_stacked[k] = torch.stack(v, dim=0) # [F, D]
        
    smplx_params_stacked['betas'] = smplx_params_stacked['betas'][0:1]

    smplx_params = dict(smplx_params_stacked)

    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0) # [1, F, D]

    smplx_params['betas'] = smplx_params['betas'].squeeze(0) 
        
    c2ws = c2ws.unsqueeze(0)       # [1, F, 4, 4]
    intrs = intrs.unsqueeze(0)     # [1, F, 4, 4]
    bg_colors = bg_colors.unsqueeze(0) # [1, F, 3]

    # print(smplx_params['betas'].shape, smplx_params['expr'].shape, smplx_params['trans'].shape, smplx_params['root_pose'].shape,smplx_params['body_pose'].shape)
    # torch.Size([1, 225, 4, 4]) torch.Size([1, 225, 4, 4]) 

    # origin: torch.Size([1, 10]) torch.Size([1, 76, 100]) torch.Size([1, 76, 3]) torch.Size([1, 76, 3]) torch.Size([1, 76, 21, 3])
    motion_seqs_ret = {
        'render_c2ws': c2ws,
        'render_intrs': intrs,
        'render_bg_colors': bg_colors,
        'smplx_params': smplx_params,
        'motion_seqs': motion_files,
    }

    return motion_seqs_ret