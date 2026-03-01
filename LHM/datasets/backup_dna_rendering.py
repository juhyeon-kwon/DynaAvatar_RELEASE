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
 
__all__ = ['DNARenderingDataset']

class DNARenderingDataset(BaseDataset):

    def __init__(self, root_dirs: list[str], meta_path: str,
                 use_flame: bool,
                 src_head_size: int,
                 n_history_length : int,
                 **data_kwargs
                 ): # qw00n; arguments from configs : **dataset_kwargs
        super().__init__(root_dirs, meta_path)

        self.n_history_length = n_history_length
        self.num_train_views = 1 # qw00n; TODO
        self.num_train_frames = 2 # qw00n; TODO
        '''
        {'0095_01': ['motion_simple'], '0047_12': ['motion_simple'], '0307_07': ['motion_simple'], '0102_02': ['interaction_hard'], '0123_02': ['interaction_hard'], '0174_09': ['interaction_hard'], '0022_10': ['motion_hard'], '0113_06': ['motion_hard'], '0018_05': ['motion_hard'], '0019_06': ['texture_simple'], '0094_02': ['texture_simple'], '0241_10': ['texture_simple'], '0124_03': ['motion_medium'], '0166_04': ['motion_medium'], '0111_08': ['motion_medium'], '0206_04': ['deformation_hard'], '0008_01': ['deformation_hard'], '0121_02': ['deformation_hard'], '0047_01': ['texture_hard'], '0097_04': ['texture_hard'], '0188_02': ['texture_hard'], '0025_11': ['deformation_medium'], '0012_09': ['deformation_medium'], '0115_07': ['deformation_medium'], '0152_01': ['interaction_no'], '0235_11': ['interaction_no'], '0307_03': ['interaction_no'], '0034_04': ['deformation_simple'], '0031_03': ['deformation_simple'], '0310_04': ['deformation_simple'], '0239_01': ['interaction_medium'], '0128_04': ['interaction_medium'], '0133_07': ['interaction_medium'], '0196_09': ['interaction_simple'], '0118_07': ['interaction_simple'], '0309_03': ['interaction_simple'], '0219_07': ['texture_medium'], '0165_08': ['texture_medium'], '0147_04': ['texture_medium']}
        '''       

        # qw00n;
        #self. ??? = data_kwargs[???] ...
        
    # qw00n; referred 4K4D repo
    def load_camera_params(self, intri_path: str, extri_path: str, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
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
            
            # --- Extrinsic Matrix ---
            Rvec = fs_extri.getNode(f'R_{camera_name}').mat()
            Tvec = fs_extri.getNode(f'T_{camera_name}').mat()

            if Rvec is not None:
                R_3x3, _ = cv2.Rodrigues(Rvec)
            else:
                R_3x3 = fs_extri.getNode(f'Rot_{camera_name}').mat()

            if R_3x3 is None or Tvec is None:
                raise ValueError(f"Rotation/Translation for '{camera_name}' not found.")
            
            # w2c -> c2w 
            R_c2w = R_3x3.T
            T_c2w = -R_3x3.T @ Tvec
            
            c2w_4x4 = np.eye(4, dtype=np.float32)
            c2w_4x4[:3, :3] = R_c2w
            c2w_4x4[:3, 3] = T_c2w.flatten()

            # (1, 4, 4) 
            intrinsic_matrix = K_4x4[None, ...]
            extrinsic_matrix = c2w_4x4[None, ...] 
            
            return intrinsic_matrix, extrinsic_matrix

        finally:
            if fs_intri is not None: fs_intri.release()
            if fs_extri is not None: fs_extri.release()
    
    def load_and_resize_image_to_tensor(self, image_path: str, max_size: int = 1024) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다. 경로: {image_path}")
            return None

        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        transform = transforms.ToTensor() 
        image_tensor = transform(image)
        return image_tensor
    
    def load_and_resize_mask_to_tensor(self, mask_path: str, target_size: tuple):
        try:
            # 'L' 모드는 흑백(grayscale) 이미지로, 채널이 1개입니다.
            mask_pil = Image.open(mask_path).convert('L')
        except FileNotFoundError:
            print(f"오류: 마스크 파일을 찾을 수 없습니다. 경로: {mask_path}")
            # 오류 발생 시, 검은색 마스크를 대신 반환합니다.
            return torch.zeros(1, target_size[0], target_size[1])

        mask_pil = mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)

        transform = transforms.ToTensor()
        mask_tensor = transform(mask_pil)  # Shape: [1, H, W]
        return mask_tensor
    
    def load_smplx(self, smplx_path: str):
        try:
            smplx_data = np.load(smplx_path)
        except FileNotFoundError:
            print(f"Error: The file '{smplx_path}' was not found.")

        # Available keys: ['scale', 'betas', 'expression', 'fullpose', 'transl']
        # betas, expression: (10,)
        # fullpose: (55, 3)
        # transl: (3,)

        return smplx_data
    
    def load_smplx_history(self, subject_path, frame_idx):
        history_smplx_poses = []
        
        current_frame_idx_int = int(frame_idx)
        for n in range(0, self.n_history_length+1):
            history_frame_idx_int = max(0, current_frame_idx_int - n)
            history_frame_idx_str = f'{history_frame_idx_int:06d}'
            smplx_path = os.path.join(subject_path, 'smplx', history_frame_idx_str + '.npz')
            
            if os.path.exists(smplx_path):
                smplx_data = self.load_smplx(smplx_path)
                pose_data = smplx_data['fullpose'] # (55, 3) 
                history_smplx_poses.append(np.vstack([pose_data, smplx_data['transl']])) # (56, 3)
            else:
                zero_pose = np.zeros((56, 3), dtype=np.float32)
                history_smplx_poses.append(zero_pose)

        smplx_pose_sequence = np.stack(history_smplx_poses, axis=0) # shape: (n_history_length + 1, 56, 3)
        return smplx_pose_sequence
        

    @no_proxy
    def inner_get_item(self, idx):
        uid = list(self.uids)[idx]
        subject_path = os.path.join(self.root_dirs, uid) # /data/DNA_Rendering/DNA_Rendering/0018_05
        
        # qw00n; need to randomly sample
        random_idx = random.randint(23, 28)
        ref_cam_idx = f'{random_idx:02d}'

        random_idx = random.randint(13, 34) # qw00n; near-front view
        tar_cam_idx = f'{random_idx:02d}'

        ref_frame_idx = '000006'

        ref_img_path = os.path.join(subject_path, 'images', ref_cam_idx, ref_frame_idx+'.jpg')
        ref_img = self.load_and_resize_image_to_tensor(ref_img_path) # (C, H, W)
        
        _, H, W = ref_img.shape
        mask_dir = os.path.join(subject_path, 'masks', ref_cam_idx)
        if os.path.exists(os.path.join(mask_dir, ref_frame_idx + '.png')):
            ref_mask_path = os.path.join(mask_dir, ref_frame_idx + '.png')
        else: # .jpg를 fallback으로 사용
            ref_mask_path = os.path.join(mask_dir, ref_frame_idx + '.jpg')
        ref_mask = self.load_and_resize_mask_to_tensor(ref_mask_path, target_size=(H, W)) # (1, H, W)
        
        # TODO
        start_sampling_idx = 10
        end_frame_idx_int = random.randint(start_sampling_idx, 149)

        gt_img_list, gt_mask_list = [], []
        target_betas_list, target_pose_list, target_transl_list = [], [], []
        motion_history_list = []

        intri_path = os.path.join(subject_path, 'intri.yml')
        extri_path = os.path.join(subject_path, 'extri.yml')
        intrinsic_matrix, extrinsic_matrix = self.load_camera_params(intri_path, extri_path, tar_cam_idx)
        
        for i in range(self.num_train_frames):
            current_frame_idx_int = end_frame_idx_int - i
            current_frame_idx_str = f'{current_frame_idx_int:06d}'

            gt_img_path = os.path.join(subject_path, 'images', tar_cam_idx, current_frame_idx_str + '.jpg')
            gt_img = self.load_and_resize_image_to_tensor(gt_img_path)
            gt_img_list.append(gt_img)

            mask_dir = os.path.join(subject_path, 'masks', tar_cam_idx)
            if os.path.exists(os.path.join(mask_dir, current_frame_idx_str + '.png')):
                mask_path = os.path.join(mask_dir, current_frame_idx_str + '.png')
            else:
                mask_path = os.path.join(mask_dir, current_frame_idx_str + '.jpg')
            gt_mask = self.load_and_resize_mask_to_tensor(mask_path, target_size=(H, W))
            gt_mask_list.append(gt_mask)

            smplx_path = os.path.join(subject_path, 'smplx', current_frame_idx_str + '.npz')
            smplx_data = self.load_smplx(smplx_path)
            
            num_betas = 10
            target_betas_list.append(torch.from_numpy(np.squeeze(smplx_data['betas'])[:num_betas]))
            target_pose_list.append(torch.from_numpy(smplx_data['fullpose']))
            target_transl_list.append(torch.from_numpy(smplx_data['transl']))

            skeleton_history = self.load_smplx_history(subject_path, current_frame_idx_str)
            motion_history_list.append(torch.from_numpy(skeleton_history))

        # 시간 순서(t-N, ..., t-1, t)로 
        gt_img_list.reverse()
        gt_mask_list.reverse()
        target_betas_list.reverse()
        target_pose_list.reverse()
        target_transl_list.reverse()
        motion_history_list.reverse()
        
        gt_imgs_tensor = torch.stack(gt_img_list, dim=0)
        gt_masks_tensor = torch.stack(gt_mask_list, dim=0)
        target_betas_tensor = torch.stack(target_betas_list, dim=0)
        target_poses_tensor = torch.stack(target_pose_list, dim=0)
        target_transls_tensor = torch.stack(target_transl_list, dim=0)
        motion_histories_tensor = torch.stack(motion_history_list, dim=0)

        return {
            'uid': uid,
            'intri': np.repeat(intrinsic_matrix, self.num_train_frames, axis=0),
            'extri': np.repeat(extrinsic_matrix, self.num_train_frames, axis=0),
            'img': ref_img.unsqueeze(0).repeat(self.num_train_frames, 1, 1, 1),                 
            'mask': ref_mask.unsqueeze(0).repeat(self.num_train_frames, 1, 1, 1),               
            
            # --- Stacked Target Data ---
            'gt_img': gt_imgs_tensor,               # shape: (num_train_frames, C, H, W)
            'gt_mask': gt_masks_tensor,             # shape: (num_train_frames, 1, H, W)
            'target_betas': target_betas_tensor,    # shape: (num_train_frames, 10)
            'target_pose': target_poses_tensor,     # shape: (num_train_frames, 55, 3)
            'target_transl': target_transls_tensor, # shape: (num_train_frames, 3)
            'motion_history': motion_histories_tensor # shape: (num_train_frames, M, 56, 3)
        }