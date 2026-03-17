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

import os
from typing import Union
import random
import numpy as np
import torch
from megfile import smart_path_join, smart_open

from .base import BaseDataset
from ..utils.proxy import no_proxy

from torchvision import transforms
from PIL import Image
import cv2
from os.path import exists

import json
import io
from collections import OrderedDict

from .toc_reader import TocReader, TocLRU
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    resize_image_keepaspect_np,
)
import math
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion


def scale_intrs(intrs: np.ndarray, ratio_x, ratio_y) -> np.ndarray:  # (B,...)
    intrs[:, 0] = intrs[:, 0] * ratio_x
    intrs[:, 1] = intrs[:, 1] * ratio_y
    return intrs

def center_crop_pad_according_to_mask(img: np.ndarray, mask: np.ndarray, aspect_standard) -> tuple[np.ndarray, np.ndarray, int, int]: 
    H, W = img.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("empty mask")
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1
    
    need_w = max(bbox_w, math.ceil(bbox_h / aspect_standard))
    need_h = max(bbox_h, math.ceil(need_w * aspect_standard))
    
    # bbox center
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    
    x0 = math.floor(cx - need_w / 2.0)
    y0 = math.floor(cy - need_h / 2.0)
    x1 = x0 + need_w
    y1 = y0 + need_h
    
    pad_left   = max(0, -x0)
    pad_top    = max(0, -y0)
    pad_right  = max(0,  x1 - W)
    pad_bottom = max(0,  y1 - H)

    crop_x0 = max(0, x0)
    crop_y0 = max(0, y0)
    crop_x1 = min(W, x1)
    crop_y1 = min(H, y1)
    
    patch_img  = img [crop_y0:crop_y1, crop_x0:crop_x1]
    patch_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]

    patch_img  = cv2.copyMakeBorder(
        patch_img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0.0
    )
    patch_mask = cv2.copyMakeBorder(
        patch_mask, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0.0
    )
    assert patch_img.shape[0] == need_h and patch_img.shape[1] == need_w, \
        (patch_img.shape, (need_h, need_w))
    #assert abs(patch_img.shape[0] / patch_img.shape[1] - aspect_standard) < 1e-6, (patch_img.shape[0], patch_img.shape[1], patch_img.shape[0] / patch_img.shape[1])
    
    offset_x = crop_x0 - pad_left
    offset_y = crop_y0 - pad_top
    
    return patch_img, patch_mask, offset_x, offset_y
 
__all__ = ['DNARenderingDataset']

# -----------------------------
# LRU (SMPLX/handle cache)
# -----------------------------
class LRU:
    def __init__(self, capacity: int = 512):
        self.cap = capacity
        self.od: "OrderedDict[object, object]" = OrderedDict()
        
    def get(self, k):
        v = self.od.get(k)
        if v is not None:
            self.od.move_to_end(k)
        return v
    
    def put(self, k, v):
        self.od[k] = v
        self.od.move_to_end(k)
        if len(self.od) > self.cap:
            self.od.popitem(last=False)

class FDDressDataset(BaseDataset):

    def __init__(self, root_dirs: list[str], meta_path: str,
                 use_flame: bool,
                 src_head_size: int,
                 n_history_length : int, 
                 fps : int,
                 smplx_lru_size: int = 256,
                 toc_lru_capacity: int = 64,
                 sqlite_mmap_size: int = 128 << 20,
                 **data_kwargs
                 ):
        super().__init__(root_dirs, meta_path)

        self.n_history_length = n_history_length
        self.num_train_views = 4
        self.num_train_frames = 4
        self.fps = fps
    
        self._toc_lru = TocLRU(self.root_dirs,
                               capacity=toc_lru_capacity,
                               sqlite_mmap_size=sqlite_mmap_size)  # worker-local: uid -> TarFile
        self._cam_params_cache = {}          # uid -> (K_4x4, c2w_4x4)
        self._smplx_lru = LRU(capacity=smplx_lru_size)         # (uid, frame) -> npz dict (LRU)
        

    # def other methods if needed   

    def _get_cam_cache(self, uid: str) -> dict: 
        if uid in self._cam_params_cache: 
            return self._cam_params_cache[uid]
        toc = self._toc_lru.get(uid)
        cam_params_b = toc.read_cam_param(uid)
        cam_params = json.loads(cam_params_b)
        if not cam_params: 
            raise FileNotFoundError(f"[{uid}] cam_params.json not found in tar")

        self._cam_params_cache[uid] = {}
        for cam_name, cam_param in cam_params.items(): 
            cam_param = {k: np.array(v) for k, v in cam_param.items()}
            # intrinsic
            K_3x3 = np.zeros((3,3))
            K_3x3[0,0], K_3x3[1,1] = cam_param['focal'][0], cam_param['focal'][1]
            K_3x3[0,2], K_3x3[1,2] = cam_param['princpt'][0], cam_param['princpt'][1]
            K_4x4 = np.eye(4, dtype=np.float32)
            K_4x4[:3, :3] = K_3x3.astype(np.float32)
            # extrinsic: w2c -> c2w 
            R_3x3 =cam_param['R']
            Tvec = cam_param['t'].reshape(3,1)
            R_c2w = R_3x3.T
            T_c2w = -R_3x3.T @ Tvec
            c2w_4x4 = np.eye(4, dtype=np.float32)
            c2w_4x4[:3, :3] = R_c2w
            c2w_4x4[:3, 3] = T_c2w.flatten()
            self._cam_params_cache[uid][cam_name] = {'K': K_4x4, 'c2w': c2w_4x4}
                
        return self._cam_params_cache[uid]
        
    def load_camera_params(self, uid: str, camera_name: str) -> tuple[np.ndarray, np.ndarray]: 
        cam_cache = self._get_cam_cache(uid)
        try: 
            
            K = cam_cache[camera_name]["K"]  # (4,4)
            c2w = cam_cache[camera_name]["c2w"]  # (4,4)
        except: 
            raise KeyError(f"[{uid}] camera parameters for cam {camera_name} not found")
        intrinsic_matrix = K[None, ...]
        extrinsic_matrix = c2w[None, ...] 
        return intrinsic_matrix, extrinsic_matrix
    
    def load_and_resize_image_to_tensor(self, uid: str, cam_idx: str, frame_idx: str, img_b: bytes, mask_b: bytes, intrinsic_matrix: np.ndarray, max_size: int = 512, return_face: bool = False) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        try:
            image = np.array(Image.open(img_b).convert('RGB'))
            image = (image / 255.).astype(np.float32)
        except: 
            print(f"Error: Image is not exists.", uid, cam_idx, frame_idx)
            return None, None, None
        height, width = image.shape[:2]
        try:
            mask = np.array(Image.open(mask_b).convert('L'))
        except:
            print(f"Error: Mask is not exists.", uid, cam_idx, frame_idx)
            mask = np.zeros((height, width), dtype=np.float32)
        mask = (mask > 0.5).astype(np.float32)
        
        # face crop
        if return_face: 
            toc = self._toc_lru.get(uid)
            face_bboxes_b = toc.read_face_bbox_bytes(uid)
            face_bboxes = json.loads(face_bboxes_b)
            try: 
                x0, y0, x1, y1 = face_bboxes[cam_idx][frame_idx]
                face_img = image[y0:y1, x0:x1, :]
                resized_head_rgb = cv2.resize(
                    face_img,
                    dsize=(128, 128),
                    interpolation=cv2.INTER_AREA,
                )
                face_bbox = np.array([x0, y0, x1, y1], dtype=np.float32)
                is_face_detected = True
            except Exception as e: 
                resized_head_rgb = np.zeros((128, 128, 3), dtype=np.float32)
                face_bbox = np.array([0, 0, 1, 1], dtype=np.float32)
                is_face_detected = False
                #print(f"No face detect: {e}")
        
        # 1. crop image to enlarge human area.
        aspect_standard = 5.0 / 3
        image_patch, mask, offset_x, offset_y = center_crop_pad_according_to_mask(
            image, mask, aspect_standard
        )
        patch_h, patch_w = image_patch.shape[:2]
        try: 
            intrinsic_matrix_rescale = intrinsic_matrix.copy()
            intrinsic_matrix_rescale[:,0, 2] -= offset_x
            intrinsic_matrix_rescale[:,1, 2] -= offset_y
        except: 
            intrinsic_matrix_rescale = None
        
        # 2. resize to render_tgt_size for training
        tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
            cur_hw=image_patch.shape[:2],
            aspect_standard=aspect_standard,
            tgt_size=max_size,
            multiply=16,
        )

        image_patch_resized = cv2.resize(image_patch, (tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA)
        if intrinsic_matrix_rescale is not None: 
            intrinsic_matrix_rescale = scale_intrs(intrinsic_matrix_rescale, ratio_x=ratio_x, ratio_y=ratio_y)

        image_tensor = torch.tensor(image_patch_resized).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask)[None,:,:]  # 1,H,W
        if return_face: 
            face_img_tensor = torch.tensor(resized_head_rgb).permute(2,0,1)  # C,H,W
            face_bbox[0], face_bbox[2] = (face_bbox[0] - offset_x) * ratio_x, (face_bbox[2] - offset_x) * ratio_x
            face_bbox[1], face_bbox[3] = (face_bbox[1] - offset_y) * ratio_y, (face_bbox[3] - offset_y) * ratio_y
            face_bbox[0] = np.clip(int(face_bbox[0]), 0, patch_w); face_bbox[1] = np.clip(int(face_bbox[1]), 0, patch_h)
            face_bbox[2] = np.clip(int(face_bbox[2]), 0, patch_w); face_bbox[3] = np.clip(int(face_bbox[3]), 0, patch_h)
            face_bbox_tensor = torch.tensor(face_bbox, dtype=torch.int64)
            return image_tensor, mask_tensor, intrinsic_matrix_rescale, face_img_tensor, face_bbox_tensor, is_face_detected
        return image_tensor, mask_tensor, intrinsic_matrix_rescale
    
    def load_image_face_to_tensor(self, uid: str, cam_idx: str, frame_idx: str, img_face_b: bytes) -> torch.Tensor:
        try:
            image_face = Image.open(img_face_b).convert('RGB')
        except: 
            print(f"w/o head input!", uid, cam_idx, frame_idx)
            image_face = Image.new('RGB', (128, 128), color=0)
        if image_face.size != (128, 128): 
            print(f'Preprocessed face image size != (128,128): {uid} {cam_idx} {frame_idx}')
            image_face = image_face.resize((128, 128), Image.Resampling.LANCZOS)

        transform = transforms.ToTensor() 
        image_tensor = transform(image_face)
        return image_tensor
    
    def load_and_resize_mask_to_tensor(self, uid: str, cam_idx: str, frame_idx: str, mask_b: bytes, target_size: tuple):
        try:
            mask_pil = Image.open(mask_b).convert('L')
        except:
            print(f"Error: Mask is not exists.", uid, cam_idx, frame_idx)
            return torch.zeros(1, target_size[0], target_size[1])

        mask_pil = mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)

        transform = transforms.ToTensor()
        mask_tensor = transform(mask_pil)  # Shape: [1, H, W]
        return mask_tensor
    
    def load_smplx(self, uid: str, frame_idx: str): 
        key = (uid, frame_idx)
        hit = self._smplx_lru.get(key)
        if hit is not None:
            return hit
        toc = self._toc_lru.get(uid)
        smplx_b, shape_param_b = toc.read_smplx_bytes(uid, frame_idx)
        try: 
            smplx_data = {}
            smplx_data['betas'] = np.array(json.loads(shape_param_b), dtype=np.float32)
            _smplx_data = json.loads(smplx_b)
            smplx_data['fullpose'] = np.concatenate([
                np.array(_smplx_data['root_pose'], dtype=np.float32).reshape(1,3), 
                np.array(_smplx_data['body_pose'], dtype=np.float32).reshape(-1,3), 
                np.array(_smplx_data['lhand_pose'], dtype=np.float32).reshape(-1,3), 
                np.array(_smplx_data['rhand_pose'], dtype=np.float32).reshape(-1,3), 
                np.array(_smplx_data['jaw_pose'], dtype=np.float32).reshape(1,3), 
                np.array(_smplx_data['leye_pose'], dtype=np.float32).reshape(1,3), 
                np.array(_smplx_data['reye_pose'], dtype=np.float32).reshape(1,3)
            ], axis=0)  # (55,3)
            smplx_data['transl'] = np.array(_smplx_data['trans'], dtype=np.float32)
            smplx_data['expr'] = np.array(_smplx_data['expr'], dtype=np.float32)
                
            self._smplx_lru.put(key, smplx_data)
        except:
            print(f"Error: The smplx data was not found.", uid, frame_idx)
            smplx_data = None
        return smplx_data
    
    def load_smplx_history(self, uid: str, frame_idx: str, sampling_stride: int, min_frame_idx: int) -> np.ndarray: 
        history_smplx_poses = []
        
        current_frame_idx_int = int(frame_idx)
        # Map world coordinate axis direction to DNA-Rendering's (+y -> -y)
        for n in range(0, self.n_history_length+1):
            history_frame_idx_int = max(min_frame_idx, current_frame_idx_int - n * sampling_stride)
            history_frame_idx_str = f'{history_frame_idx_int:06d}'
            smplx_data = self.load_smplx(uid, history_frame_idx_str)
            
            if smplx_data is None: 
                history_smplx_poses.append(history_smplx_poses[-1].copy())
            else: 
                rot_4d_to_dna = torch.tensor([[1.,0.,0.],
                                    [0.,-1.,0.],
                                    [0.,0.,-1.]], dtype=torch.float32)
                pose_data = smplx_data['fullpose'].copy()
                root_pose_orig = axis_angle_to_matrix(torch.tensor(pose_data[0,:]))
                pose_data[0] = np.array(matrix_to_axis_angle(rot_4d_to_dna @ root_pose_orig))
                trans = np.array(rot_4d_to_dna @ torch.tensor(smplx_data['transl'])[:,None]).reshape(3)
                history_smplx_poses.append(np.vstack([pose_data, trans]))  # (56,3)

        smplx_pose_sequence = np.stack(history_smplx_poses, axis=0) # shape: (n_history_length + 1, 56, 3)
        return smplx_pose_sequence
        

    @no_proxy
    def inner_get_item(self, idx):
        uid = list(self.uids)[idx]
        subject_path = os.path.join(self.root_dirs, uid)
        
        toc = self._toc_lru.get(uid)
        
        # 1. Load Reference Data (View-Independent)
        face_bboxes_b = toc.read_face_bbox_bytes(uid)
        face_bboxes = json.loads(face_bboxes_b)
        face_bboxes_cam_idx_list = tuple(face_bboxes.keys())  # TODO: caching?
        if not face_bboxes_cam_idx_list:
            raise RuntimeError(f"{uid}: face_bboxes is empty")
        ref_cam_idx = random.choice(face_bboxes_cam_idx_list)
        
        # get min/max frame idx
        min_frame_idx, max_frame_idx = toc.get_min_max_frame_idx(uid, None)
        if min_frame_idx is None or max_frame_idx is None:
            raise RuntimeError(f"{uid}: has no frames found in images table")
        
        ref_frame_idx = random.choice(tuple(face_bboxes[ref_cam_idx].keys()))

        # Load & resize ref img, mask, and cam_param
        ref_img_b = toc.read_image_bytes(uid, ref_cam_idx, ref_frame_idx)
        ref_mask_b = toc.read_mask_bytes(uid, ref_cam_idx, ref_frame_idx)
        ref_img, ref_mask, _, ref_face_img, _, is_ref_face_detected = self.load_and_resize_image_to_tensor(
            uid, ref_cam_idx, ref_frame_idx, 
            io.BytesIO(ref_img_b), io.BytesIO(ref_mask_b), 
            None,
            return_face=True
        ) 
        
        # 2. Sample Target Cameras (num_train_views)
        tar_cam_idx_list = []
        for _ in range(self.num_train_views):
            random_idx = random.randint(0, 23)
            tar_cam_idx_list.append(f'{random_idx:02d}')

        # 3. Prepare Frame Indices
        sampling_stride = max(self.fps // 15, 1)
        assert sampling_stride == 2 

        frame_idx_int_list = []
        for i in range(self.num_train_frames):
            current_frame_idx_int = random.randint(int(min_frame_idx), int(max_frame_idx))
            frame_idx_int_list.append(current_frame_idx_int)
        
        # View-dependent data (Frame-major list: [num_train_frames] x [num_train_views] x ...)
        all_frames_gt_img_list = []
        all_frames_gt_mask_list = []
        all_frames_intri_list = []
        all_frames_extri_list = []
        all_frames_target_face_bbox_list = []
        all_frames_is_target_face_detected_list = []
        
        # View-independent data (Frame-major list: [num_train_frames] x ...)
        target_betas_list = []
        target_exprs_list = []
        target_pose_list = []
        target_transl_list = []
        motion_history_list = []

        # 5. Main Data Loading Loop
        
        # Outer loop: Frames
        for current_frame_idx_int in frame_idx_int_list:
            current_frame_idx_str = f'{current_frame_idx_int:06d}'
            
            # Load View-Independent Data (Per-Frame)
            smplx_data = self.load_smplx(uid, current_frame_idx_str)
            
            num_betas = 10
            target_betas_list.append(torch.from_numpy(np.squeeze(smplx_data['betas'])[:num_betas]))
            target_exprs_list.append(torch.from_numpy(np.squeeze(smplx_data['expr'])[:num_betas]))
            target_pose_list.append(torch.from_numpy(smplx_data['fullpose']))
            target_transl_list.append(torch.from_numpy(smplx_data['transl']))
            
            skeleton_history = self.load_smplx_history(uid, current_frame_idx_str, sampling_stride, int(min_frame_idx))
            motion_history_list.append(torch.from_numpy(skeleton_history))

            # Inner loop: Views
            current_frame_gt_img_list = []
            current_frame_gt_mask_list = []
            current_frame_intri_list = []
            current_frame_extri_list = []
            current_frame_target_face_bbox_list = []
            current_frame_is_target_face_detected_list = []
            
            for tar_cam_idx in tar_cam_idx_list:
                # Load cam param (Per-View)
                intrinsic_matrix, extrinsic_matrix = self.load_camera_params(uid, tar_cam_idx)  # (1,4,4), (1,4,4)

                # Load image & mask (Per-Frame, Per-View)
                gt_img_b = toc.read_image_bytes(uid, tar_cam_idx, current_frame_idx_str)
                gt_mask_b = toc.read_mask_bytes(uid, tar_cam_idx, current_frame_idx_str)
                
                gt_img, gt_mask, gt_cam_param, _, target_face_bbox, is_target_face_detected = self.load_and_resize_image_to_tensor(
                    uid, tar_cam_idx, current_frame_idx_str, 
                    io.BytesIO(gt_img_b), io.BytesIO(gt_mask_b), 
                    intrinsic_matrix,
                    return_face=True
                )
                
                current_frame_gt_img_list.append(gt_img)
                current_frame_gt_mask_list.append(gt_mask)
                current_frame_intri_list.append(gt_cam_param) # (1, 4, 4)
                current_frame_extri_list.append(extrinsic_matrix) # (1, 4, 4)
                current_frame_target_face_bbox_list.append(target_face_bbox)
                current_frame_is_target_face_detected_list.append(is_target_face_detected)

            all_frames_gt_img_list.append(torch.stack(current_frame_gt_img_list, dim=0))
            all_frames_gt_mask_list.append(torch.stack(current_frame_gt_mask_list, dim=0))
            all_frames_intri_list.append(np.concatenate(current_frame_intri_list, axis=0)) # (num_train_views, 4, 4)
            all_frames_extri_list.append(np.concatenate(current_frame_extri_list, axis=0)) # (num_train_views, 4, 4)
            all_frames_target_face_bbox_list.append(torch.stack(current_frame_target_face_bbox_list, dim=0))
            all_frames_is_target_face_detected_list.append(torch.tensor(current_frame_is_target_face_detected_list))

        # 6. Final Stacking
        
        # Stack view-dependent data (dim=0) -> (num_train_frames, num_train_views, ...)
        gt_imgs_tensor = torch.stack(all_frames_gt_img_list, dim=0)
        gt_masks_tensor = torch.stack(all_frames_gt_mask_list, dim=0)
        intri_tensor = np.stack(all_frames_intri_list, axis=0)
        extri_tensor = np.stack(all_frames_extri_list, axis=0)
        target_face_bboxes_tensor = torch.stack(all_frames_target_face_bbox_list, dim=0)
        is_target_face_detected_tensor = torch.stack(all_frames_is_target_face_detected_list, dim=0)

        # Stack view-independent data (dim=0) -> (num_train_frames, ...)
        target_betas_tensor = torch.stack(target_betas_list, dim=0)
        target_exprs_tensor = torch.stack(target_exprs_list, dim=0)
        target_poses_tensor = torch.stack(target_pose_list, dim=0)
        target_transls_tensor = torch.stack(target_transl_list, dim=0)
        motion_histories_tensor = torch.stack(motion_history_list, dim=0)

        # 7. Expand Reference Data to Match Output Shape
        # (C, H, W) -> (1, 1, C, H, W) -> (F, 1, C, H, W)
        ref_img_expanded = ref_img.unsqueeze(0).unsqueeze(0).repeat(
            self.num_train_frames, 1, 1, 1, 1
        )
        ref_face_img_expanded = ref_face_img.unsqueeze(0).unsqueeze(0).repeat(
            self.num_train_frames, 1, 1, 1, 1
        )
        # (1, H, W) -> (1, 1, 1, H, W) -> (F, 1, 1, H, W)
        ref_mask_expanded = ref_mask.unsqueeze(0).unsqueeze(0).repeat(
            self.num_train_frames, 1, 1, 1, 1
        )
        # bool -> (1, 1) -> (F, 1)
        is_ref_face_detected_expanded = torch.tensor([is_ref_face_detected]).unsqueeze(0).repeat(
            self.num_train_frames, 1
        )
        
        return {
            'uid': uid,
            
            # --- View-Dependent Target Data ---
            'intri': intri_tensor,                  # shape: (F, V, 4, 4)
            'extri': extri_tensor,                  # shape: (F, V, 4, 4)
            'gt_img': gt_imgs_tensor,               # shape: (F, V, C, H, W)
            'gt_mask': gt_masks_tensor,             # shape: (F, V, 1, H, W)
            'is_target_face_detected': is_target_face_detected_tensor,  # shape: (F, V)
            'target_face_bboxes_tensor': target_face_bboxes_tensor,     # shape: (F, V, 4)

            # --- View-Independent Target Data ---
            'target_betas': target_betas_tensor,    # shape: (F, 10)
            'target_exprs': target_exprs_tensor,
            'target_pose': target_poses_tensor,     # shape: (F, 55, 3)
            'target_transl': target_transls_tensor, # shape: (F, 3)
            'motion_history': motion_histories_tensor, # shape: (F, M, 56, 3)
            
            # --- Reference Data (Expanded) ---
            'img': ref_img_expanded,                # shape: (F, 1, C, H, W)
            'face_img': ref_face_img_expanded,      # shape: (F, 1, C, H, W)
            'is_ref_face_detected': is_ref_face_detected_expanded, # shape: (F, 1)
            'mask': ref_mask_expanded,              # shape: (F, 1, 1, H, W)
        }
    
    
    def close(self):
        try: 
            self._toc_lru.close_all()
        except Exception: 
            pass

    def __del__(self): 
        self.close()