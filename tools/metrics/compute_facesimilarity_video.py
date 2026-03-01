# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-03 10:29:00
# @Function      : easy to use FaceSimilarity metric

import os
import pdb
import shutil
import sys

sys.path.append("./")
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from prettytable import PrettyTable
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm

from openlrm.models.arcface_utils import ResNetArcFace
from openlrm.utils.face_detector import FaceDetector

device = "cuda"
model_path = "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd"
face_detector = FaceDetector(model_path=model_path, device=device)

id_face_net = ResNetArcFace()
id_face_net.cuda()
id_face_net.eval()


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


def write_json(path, x):
    """write a json file.

    Args:
        path (str): path to write json file.
        x (dict): dict to write.
    """
    import json

    with open(path, "w") as f:
        json.dump(x, f, indent=2)


def crop_face_image(image_path):
    rgb = np.array(Image.open(image_path))
    rgb = torch.from_numpy(rgb).permute(2, 0, 1)
    bbox = face_detector(rgb)
    head_rgb = rgb[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
    head_rgb = head_rgb.permute(1, 2, 0)
    head_rgb = head_rgb.cpu().numpy()
    return head_rgb


def gray_resize_for_identity(out, size=128):
    out_gray = (
        0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :]
    )
    out_gray = out_gray.unsqueeze(1)
    out_gray = F.interpolate(
        out_gray, (size, size), mode="bilinear", align_corners=False
    )
    return out_gray


@torch.no_grad()
def eval(input_folder, target_folder, device="cuda"):

    gt_imgs = get_image_paths_current_dir(target_folder)
    result_imgs = get_image_paths_current_dir(input_folder)

    if "visualization" in result_imgs[-1]:
        result_imgs = result_imgs[:-1]

    if len(gt_imgs) != len(result_imgs):
        return -1

    to_tensor = transforms.ToTensor()

    face_id_loss_list = []

    for input_img, gt_img in zip(result_imgs, gt_imgs):

        try:
            input_img = crop_face_image(input_img)
            input_head_tensor = gray_resize_for_identity(
                to_tensor(input_img).unsqueeze(0).to(device)
            )
            input_head_feature = id_face_net(input_head_tensor).detach()

            head_img = crop_face_image(gt_img)
            head_img = to_tensor(head_img).unsqueeze(0).to(device)
            src_head_tensor = gray_resize_for_identity(head_img)
            src_head_feature = id_face_net(src_head_tensor).detach()

            face_id_loss = F.l1_loss(input_head_feature, src_head_feature)
            face_id_loss_list.append(face_id_loss.item())
        except:
            continue

    if len(face_id_loss_list) > 0:

        return np.mean(face_id_loss_list)  # return max similarity view.
    else:
        return -1


def get_parse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--folder1", type=str, required=True)
    parser.add_argument("-f2", "--folder2", type=str, required=True)
    parser.add_argument("--pad", action="store_true")
    parser.add_argument("--pre", default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    opt = get_parse()

    input_folder = opt.folder1
    target_folder = opt.folder2

    valid_txt = os.path.join(input_folder, "front_view.txt")

    target_folder = target_folder[:-1] if target_folder[-1] == "/" else target_folder

    target_key = target_folder.split("/")[-2:]

    save_folder = os.path.join(f"./exps/metrics{opt.pre}", "psnr_results", *target_key)
    os.makedirs(save_folder, exist_ok=True)

    with open(valid_txt) as f:
        items = f.read().splitlines()
        items = [x.split(" ")[0] for x in items]

    results_dict = defaultdict(dict)
    face_similarity_list = []

    for item in tqdm(items):

        target_item_folder = os.path.join(input_folder, item)
        input_item_folder = os.path.join(target_folder, item, "rgb")

        if os.path.exists(input_item_folder) and os.path.exists(target_item_folder):

            fs_ = eval(input_item_folder, target_item_folder)

            if fs_ == -1:
                continue

            face_similarity_list.append(fs_)

            results_dict[item]["face_similarity"] = fs_
            if opt.debug:
                break
            print(results_dict)

    results_dict["all_mean"]["face_similarity"] = np.mean(face_similarity_list)

    write_json(os.path.join(save_folder, "face_similarity.json"), results_dict)
