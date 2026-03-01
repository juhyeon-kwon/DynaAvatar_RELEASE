# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-03 10:28:47
# @Function      : easy to use SSIM and LPIPS metric

import os
import pdb
import shutil
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from prettytable import PrettyTable
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
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


def img_center_padding(img_np, pad_ratio=0.2):

    ori_w, ori_h = img_np.shape[:2]

    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)

    img_pad_np = (np.ones((w, h, 3), dtype=img_np.dtype) * 255).astype(np.uint8)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[
        offset_h : offset_h + img_np.shape[0] :, offset_w : offset_w + img_np.shape[1]
    ] = img_np

    return img_pad_np, offset_w, offset_h


def scan_files_in_dir(directory, postfix=None, progress_bar=None) -> list:
    file_list = []
    progress_bar = (
        tqdm(total=0, desc=f"Scanning", ncols=100)
        if progress_bar is None
        else progress_bar
    )
    for entry in os.scandir(directory):
        if entry.is_file():
            if postfix is None or os.path.splitext(entry.path)[1] in postfix:
                file_list.append(entry)
                progress_bar.total += 1
                progress_bar.update(1)
        elif entry.is_dir():
            file_list += scan_files_in_dir(
                entry.path, postfix=postfix, progress_bar=progress_bar
            )
    return file_list


class EvalDataset(Dataset):
    def __init__(self, gt_folder, pred_folder, height=1024):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.height = height
        self.data = self.prepare_data()
        self.to_tensor = transforms.ToTensor()

    def extract_id_from_filename(self, filename):
        # find first number in filename
        start_i = None
        for i, c in enumerate(filename):
            if c.isdigit():
                start_i = i
                break
        if start_i is None:
            assert False, f"Cannot find number in filename {filename}"
        return filename[start_i : start_i + 8]

    def prepare_data(self):
        gt_files = scan_files_in_dir(self.gt_folder, postfix={".jpg", ".png"})

        gt_dict = {self.extract_id_from_filename(file.name): file for file in gt_files}
        pred_files = scan_files_in_dir(self.pred_folder, postfix={".jpg", ".png"})

        pred_files = list(filter(lambda x: "visualization" not in x.name, pred_files))

        tuples = []
        for pred_file in pred_files:
            pred_id = self.extract_id_from_filename(pred_file.name)
            if pred_id not in gt_dict:
                print(f"Cannot find gt file for {pred_file}")
            else:
                tuples.append((gt_dict[pred_id].path, pred_file.path))
        return tuples

    def resize(self, img):
        w, h = img.size
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.LANCZOS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, pred_path = self.data[idx]

        gt, pred = self.resize(Image.open(gt_path)), self.resize(Image.open(pred_path))
        if gt.height != self.height:
            gt = self.resize(gt)
        if pred.height != self.height:
            pred = self.resize(pred)
        gt = self.to_tensor(gt)
        pred = self.to_tensor(pred)
        return gt, pred


def copy_resize_gt(gt_folder, height):
    new_folder = os.path.join(
        os.path.dirname(gt_folder[:-1] if gt_folder[-1] == "/" else gt_folder),
        f"resize_{height}",
    )
    if not os.path.exists(new_folder):
        os.makedirs(new_folder, exist_ok=True)
    for file in tqdm(os.listdir(gt_folder)):
        if os.path.exists(os.path.join(new_folder, file)):
            continue
        img = Image.open(os.path.join(gt_folder, file))
        img = np.asarray(img)
        # img, _, _ = img_center_padding(img)
        img = Image.fromarray(img)
        w, h = img.size
        img.save(os.path.join(new_folder, file))
    return new_folder


@torch.no_grad()
def ssim(dataloader):
    ssim_score = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        ssim_score += ssim(pred, gt) * batch_size
    return ssim_score / len(dataloader.dataset)


@torch.no_grad()
def lpips(dataloader):
    lpips_score = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to("cuda")
    score = 0
    for gt, pred in tqdm(dataloader, desc="Calculating LPIPS"):
        batch_size = gt.size(0)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        # LPIPS needs the images to be in the [-1, 1] range.
        gt = (gt * 2) - 1
        pred = (pred * 2) - 1
        score += lpips_score(gt, pred) * batch_size
    return score / len(dataloader.dataset)


def eval(pred_folder, gt_folder):
    # Check gt_folder has images with target height, resize if not
    pred_sample = os.listdir(pred_folder)[0]
    gt_sample = os.listdir(gt_folder)[0]

    img = Image.open(os.path.join(pred_folder, pred_sample))
    gt_img = Image.open(os.path.join(gt_folder, gt_sample))

    copy_folder = None
    if img.height != gt_img.height:
        title = "--" * 30 + "Resizing GT Images to height {img.height}" + "--" * 30
        print(title)
        gt_folder = copy_resize_gt(gt_folder, img.height)
        print("-" * len(title))
        copy_folder = gt_folder

    # Form dataset
    dataset = EvalDataset(gt_folder, pred_folder, img.height)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    # Calculate Metrics
    header = []
    row = []

    header += ["SSIM", "LPIPS"]
    ssim_ = ssim(dataloader).item()
    lpips_ = lpips(dataloader).item()
    row += [ssim_, lpips_]

    # Print Results
    print("GT Folder  : ", gt_folder)
    print("Pred Folder: ", pred_folder)
    table = PrettyTable()
    table.field_names = header
    table.add_row(row)

    if copy_folder is not None:
        shutil.rmtree(copy_folder)

    return ssim_, lpips_


def get_parse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--folder1", type=str, required=True)
    parser.add_argument("-f2", "--folder2", type=str, required=True)
    parser.add_argument("--pre", type=str, default="anigs")
    parser.add_argument("--pad", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    opt = get_parse()

    input_folder = opt.folder1
    target_folder = opt.folder2

    save_folder = os.path.join(
        f"./exps/metrics{opt.pre}", "psnr_results", "anigs_video"
    )
    os.makedirs(save_folder, exist_ok=True)

    input_folders = tu.next_folders(input_folder)

    results_dict = defaultdict(dict)
    lpips_list = []
    ssim_list = []

    for input_folder in input_folders:

        item_basename = tu.basename(input_folder)

        mask_item_folder = None
        input_item_folder = os.path.join(input_folder, "rgb")
        target_item_folder = os.path.join(target_folder, item_basename)

        if os.path.exists(input_item_folder) and os.path.exists(target_item_folder):

            ssim_, lpips_ = eval(input_item_folder, target_item_folder)

            if ssim_ == -1:
                continue

            lpips_list.append(lpips_)
            ssim_list.append(ssim_)

            results_dict[item_basename]["lpips"] = lpips_
            results_dict[item_basename]["ssim"] = ssim_
            if opt.debug:
                break
            print(results_dict)

    results_dict["all_mean"]["lpips"] = np.mean(lpips_list)
    results_dict["all_mean"]["ssim"] = np.mean(ssim_list)

    write_json(os.path.join(save_folder, "lpips_ssim.json"), results_dict)
