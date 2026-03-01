# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-03 10:28:35
# @Function      : Easy to use PSNR metric
import os
import sys

sys.path.append("./")

import math
import pdb

import cv2
import numpy as np
import skimage
import torch
from PIL import Image
from tqdm import tqdm
from tqlt import utils as tu


def write_json(path, x):
    """write a json file.

    Args:
        path (str): path to write json file.
        x (dict): dict to write.
    """
    import json

    with open(path, "w") as f:
        json.dump(x, f, indent=2)


def img_center_padding(img_np, pad_ratio=0.2, background=1):

    ori_w, ori_h = img_np.shape[:2]

    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)

    if background == 1:
        img_pad_np = np.ones((w, h, 3), dtype=img_np.dtype)
    else:
        img_pad_np = np.zeros((w, h, 3), dtype=img_np.dtype)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[
        offset_h : offset_h + img_np.shape[0] :, offset_w : offset_w + img_np.shape[1]
    ] = img_np

    return img_pad_np, offset_w, offset_h


def compute_psnr(src, tar):
    psnr = skimage.metrics.peak_signal_noise_ratio(tar, src, data_range=1)
    return psnr


def get_parse():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f1", "--folder1", required=True, help="input path")
    parser.add_argument("-f2", "--folder2", required=True, help="output path")
    parser.add_argument("-m", "--mask", default=None, help="output path")
    parser.add_argument("--pre", default="anigs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pad", action="store_true")
    args = parser.parse_args()
    return args


def get_image_paths_current_dir(folder_path):
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".jfif",
    }

    return sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
    )


def psnr_compute(
    input_data,
    results_data,
    mask_data=None,
    pad=False,
):

    gt_imgs = get_image_paths_current_dir(input_data)
    result_imgs = get_image_paths_current_dir(os.path.join(results_data))

    if mask_data is not None:
        mask_imgs = get_image_paths_current_dir(mask_data)
    else:
        mask_imgs = None

    if "visualization" in result_imgs[-1]:
        result_imgs = result_imgs[:-1]

    if len(gt_imgs) != len(result_imgs):
        return -1

    gt_imgs = gt_imgs[::4]
    result_imgs = result_imgs[::4]

    psnr_mean = []

    for mask_i, (gt, result) in tqdm(enumerate(zip(gt_imgs, result_imgs))):
        result_img = (cv2.imread(result, cv2.IMREAD_UNCHANGED) / 255.0).astype(
            np.float32
        )
        gt_img = (cv2.imread(gt, cv2.IMREAD_UNCHANGED) / 255.0).astype(np.float32)

        if mask_imgs is not None:
            mask_img = (
                cv2.imread(mask_imgs[mask_i], cv2.IMREAD_UNCHANGED) / 255.0
            ).astype(np.float32)
            mask_img = mask_img[..., -1]
            mask_img = np.stack([mask_img] * 3, axis=-1)
            mask_img, _, _ = img_center_padding(mask_img, background=0)

        if pad:
            gt_img, _, _ = img_center_padding(gt_img)

        h, w, c = result_img.shape

        gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_AREA)
        if mask_imgs is not None:
            mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_AREA)
            gt_img = gt_img * mask_img + 1 - mask_img
            result_img = result_img * mask_img + 1 - mask_img
            mask_label = mask_img[..., 0]
            psnr_mean += [
                compute_psnr(result_img[mask_label > 0.5], gt_img[mask_label > 0.5])
            ]
        else:
            psnr_mean += [compute_psnr(result_img, gt_img)]

        # Image.fromarray((gt_img * 255).astype(np.uint8)).save("gt.png")
        # Image.fromarray((result_img * 255).astype(np.uint8)).save("result.png")

    psnr = np.mean(psnr_mean)

    return psnr


if __name__ == "__main__":

    opt = get_parse()

    input_folder = opt.folder1
    target_folder = opt.folder2
    mask_folder = opt.mask

    save_folder = os.path.join(
        f"./exps/metrics{opt.pre}", "psnr_results", "anigs_video"
    )
    os.makedirs(save_folder, exist_ok=True)

    input_folders = tu.next_folders(input_folder)

    results_dict = dict()
    psnr_list = []

    for input_folder in input_folders:

        item_basename = tu.basename(input_folder)

        mask_item_folder = None
        input_item_folder = os.path.join(input_folder, "rgb")
        target_item_folder = os.path.join(target_folder, item_basename)

        if os.path.exists(input_item_folder) and os.path.exists(target_item_folder):

            psnr = psnr_compute(
                input_item_folder, target_item_folder, mask_item_folder, opt.pad
            )

            if psnr == -1:
                continue

            psnr_list.append(psnr)

            results_dict[item_basename] = psnr
            if opt.debug:
                break
            print(results_dict)

    results_dict["all_mean"] = np.mean(psnr_list)
    write_json(os.path.join(save_folder, "PSNR.json"), results_dict)
