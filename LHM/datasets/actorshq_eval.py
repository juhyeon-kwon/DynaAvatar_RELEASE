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

#### jane;
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
from tqdm import tqdm

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
    
    '''(주석 처리된 코드 생략)'''

    # 이미지 바깥으로 나간 만큼 패딩 필요량
    pad_left   = max(0, -x0)
    pad_top    = max(0, -y0)
    pad_right  = max(0,  x1 - W)
    pad_bottom = max(0,  y1 - H)

    # 실제로 잘라낼 영역(이미지 내부 교집합)
    crop_x0 = max(0, x0)
    crop_y0 = max(0, y0)
    crop_x1 = min(W, x1)
    crop_y1 = min(H, y1)
    
    patch_img  = img [crop_y0:crop_y1, crop_x0:crop_x1]
    patch_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]

    # 패딩으로 목표 크기 정확히 맞추기 (좌/상 우/하)
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
 
__all__ = ['ActorsHQDataset']

# -----------------------------
# LRU (SMPLX/핸들 캐시에 사용)
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







class ActorsHQDataset(BaseDataset):

    def __init__(self, root_dirs: list[str], meta_path: str,
                 use_flame: bool,
                 src_head_size: int,
                 n_history_length : int, 
                 fps : int,
                 save_path : str,

                 num_train_frames: int = 1, # 👈 qw00n; batch size

                 smplx_lru_size: int = 256,   # jane;
                 toc_lru_capacity: int = 64,   # jane;
                 sqlite_mmap_size: int = 128 << 20,   # jane;
                 **data_kwargs
                 ): # qw00n; arguments from configs : **dataset_kwargs
        super().__init__(root_dirs, meta_path)

        self.n_history_length = n_history_length
        self.num_train_frames = num_train_frames # 👈 
        self.fps = fps
        self.save_path = save_path

        # jane;
        # I/O 객체들을 여기에서 초기화하지 않습니다. 대신, 각 워커에서 필요할 때 초기화하도록 None으로 설정합니다.
        self._toc_lru_local = None
        self._cam_params_cache_local = None
        self._smplx_lru_local = None

        # 워커-로컬 초기화를 위한 설정값 저장
        self._toc_lru_capacity = toc_lru_capacity
        self._sqlite_mmap_size = sqlite_mmap_size
        self._smplx_lru_size = smplx_lru_size

        self.samples = []
        self.ref_data_cache = {}
        self.all_cam_params_cache = {} # 👈 
        
        max_cam_id = 160 # 👈 dataset 종류 따라 넉넉하게만 잡으면 됨. 이만큼 for문 돌면서 없으면 camera append 안하고 pass하는 방식
        self.sampling_stride = max(self.fps // 15, 1) # 👈 
        
        # __init__은 메인 프로세스에서 한 번만 실행됩니다. 샘플 목록과 ref_data_cache를 빌드하기 위해 *임시* TocLRU를 생성합니다.
        init_toc_lru = TocLRU(self.root_dirs,
                              capacity=toc_lru_capacity,
                              sqlite_mmap_size=sqlite_mmap_size)

        print(f"[{self.__class__.__name__}] Building 'eval' mode samples...")
        for uid in tqdm(self.uids):
            try:
                # 임시 toc_lru 사용
                subject_id = uid.split('_')[0]
                ref_uid = f"{subject_id}_1"
                toc_ref = init_toc_lru.get(ref_uid)
                
                # 1. 이 UID의 고정 참조(Reference) 데이터 로드 및 캐시
                face_bboxes_b = toc_ref.read_face_bbox_bytes(ref_uid)
                face_bboxes = json.loads(face_bboxes_b)
                face_bboxes_cam_idx_list = tuple(face_bboxes.keys())
                if not face_bboxes_cam_idx_list:
                    print(f"Warning: No face bboxes for {uid}, skipping.")
                    continue
                    
                # 예: 첫 번째 유효한 카메라와 프레임을 참조로 고정
                ref_cam_idx = '127'
                ref_frame_idx_list = list(face_bboxes[ref_cam_idx].keys())
                ref_frame_idx_list.sort(key=lambda x: int(x))
                ref_frame_idx = ref_frame_idx_list[0]
                ref_img_b = toc_ref.read_image_bytes(ref_uid, ref_cam_idx, ref_frame_idx)
                ref_mask_b = toc_ref.read_mask_bytes(ref_uid, ref_cam_idx, ref_frame_idx)
                
                # --- (수정) ---
                # load_and_resize_image_to_tensor에 임시 toc 객체 전달
                ref_img, ref_mask, _, ref_face_img, _, is_ref_face_detected = self.load_and_resize_image_to_tensor(
                    toc_ref, ref_uid, ref_cam_idx, ref_frame_idx, # toc 객체 전달
                    io.BytesIO(ref_img_b), io.BytesIO(ref_mask_b),
                    None, return_face=True
                )
                # --- (수정 끝) ---
                
                self.ref_data_cache[uid] = {
                    'img': ref_img, 'mask': ref_mask, 'face_img': ref_face_img,
                    'is_ref_face_detected': torch.tensor([is_ref_face_detected])
                }
            except Exception as e: 
                print("Error handling ", uid, " message: ", e)

            # 👈 모든 카메라 파라미터 로드 및 캐시
            toc = init_toc_lru.get(uid)
            cam_params_b = toc.read_cam_param(uid)
            cam_params = json.loads(cam_params_b)
            if not cam_params:
                print(f"Warning: No cam_params.json for {uid}, skipping.")
                continue

            temp_cam_cache = {}
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
                temp_cam_cache[cam_name] = {'K': K_4x4, 'c2w': c2w_4x4}

            all_intri_list = []
            all_extri_list = []
            all_h_list=[]
            all_w_list=[]
            for i in range(max_cam_id + 1):
                # ActorsHQ 카메라는 000, 001, ... 160 형식
                cam_name = f"{i:03d}" 
                
                if cam_name in temp_cam_cache:
                    all_intri_list.append(temp_cam_cache[cam_name]['K'])
                    all_extri_list.append(temp_cam_cache[cam_name]['c2w'])

                    ex_img_b = toc.read_image_bytes(uid, cam_name, ref_frame_idx)
                    ex_image = np.array(Image.open(io.BytesIO(ex_img_b)).convert('RGB'))
                    height, width = ex_image.shape[:2]

                    all_h_list.append(height)
                    all_w_list.append(width)

                    # render 저장될 폴더 미리 생성
                    save_path = os.path.join(self.save_path, uid, cam_name)
                    os.makedirs(save_path, exist_ok=True)
                else:
                    pass 

            # ActorsHQ는 카메라 수가 가변적이므로, 존재하는 것만 stack
            self.all_cam_params_cache[uid] = {
                'intri': np.stack(all_intri_list, axis=0)[None, ...], # (1, V, 4, 4)
                'extri': np.stack(all_extri_list, axis=0)[None, ...],  # (1, V, 4, 4)
                'height': np.stack(all_h_list, axis=0)[None, ...],
                'width': np.stack(all_w_list, axis=0)[None, ...]
            }

            min_frame, max_frame = toc.get_min_max_frame_idx(uid, None) #
            if min_frame is None:
                continue
            
            min_frame_int = int(min_frame) + 45 # qw00n; actorshq min frame is 45
            max_frame_int = int(max_frame)

            # 청크를 구성할 수 있는 가장 이른 "끝 프레임"
            first_valid_end_frame = min_frame_int + (self.num_train_frames - 1) 
            if first_valid_end_frame > max_frame_int:
                continue # 이 UID는 프레임이 충분하지 않음
            

            # 👈  frame clipping 
            eval_chunk_skip = 15 # qw00n; 
            for end_frame_int in range(first_valid_end_frame, max_frame_int + 1, eval_chunk_skip):
                self.samples.append((uid, end_frame_int))

            
        # 임시 TocLRU 핸들 닫기 및 삭제
        init_toc_lru.close_all()
        del init_toc_lru

        print(f"[{self.__class__.__name__}] Created {len(self.samples)} evaluation samples (chunks).") # 👈 **[수정]**

    # 워커-로컬(worker-local) 캐시 객체들을 위한 프로퍼티(getter)
    # 각 워커에서 이 프로퍼티에 처음 접근할 때 객체가 초기화됩니다.
    @property
    def toc_lru(self) -> TocLRU:
        if self._toc_lru_local is None:
            self._toc_lru_local = TocLRU(self.root_dirs,
                                         capacity=self._toc_lru_capacity,
                                         sqlite_mmap_size=self._sqlite_mmap_size)
        return self._toc_lru_local

    @property
    def smplx_lru(self) -> LRU:
        if self._smplx_lru_local is None:
            self._smplx_lru_local = LRU(capacity=self._smplx_lru_size)
        return self._smplx_lru_local

    @property
    def cam_params_cache(self) -> dict:
        if self._cam_params_cache_local is None:
            self._cam_params_cache_local = {}
        return self._cam_params_cache_local

    def __len__(self):
        return len(self.samples) # 👈 전체 (uid, end_frame) 청크 샘플 수

    # jane;
    def _get_cam_cache(self, uid: str) -> dict: 
        # 워커-로컬 프로퍼티 사용
        if uid in self.cam_params_cache: 
            return self.cam_params_cache[uid]
        toc = self.toc_lru.get(uid)
        
        cam_params_b = toc.read_cam_param(uid)
        cam_params = json.loads(cam_params_b)
        if not cam_params: 
            raise FileNotFoundError(f"[{uid}] cam_params.json not found in tar")
        
        self.cam_params_cache[uid] = {} # 👈
        
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
            
            self.cam_params_cache[uid][cam_name] = {'K': K_4x4, 'c2w': c2w_4x4}
        
        return self.cam_params_cache[uid]
        
    def load_camera_params(self, uid: str, camera_name: str) -> tuple[np.ndarray, np.ndarray]: 
        # [참고] 👈 이 함수는 FDDressDataset와 마찬가지로 inner_get_item에서 더 이상 사용되지 않습니다.
        cam_cache = self._get_cam_cache(uid)
        try: 
            
            K = cam_cache[camera_name]["K"]  # (4,4)
            c2w = cam_cache[camera_name]["c2w"]  # (4,4)
        except: 
            raise KeyError(f"[{uid}] camera parameters for cam {camera_name} not found")
        intrinsic_matrix = K[None, ...]
        extrinsic_matrix = c2w[None, ...] 
        return intrinsic_matrix, extrinsic_matrix
    
    # 👈 toc_reader를 첫 번째 인자로 받도록 시그니처 변경
    def load_and_resize_image_to_tensor(self, toc: TocReader, uid: str, cam_idx: str, frame_idx: str, img_b: bytes, mask_b: bytes, intrinsic_matrix: np.ndarray, max_size: int = 512, return_face: bool = False) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        try:
            image = np.array(Image.open(img_b).convert('RGB'))  # H,W,C
            image = (image / 255.).astype(np.float32)
        except: 
            print(f"오류: 이미지 데이터가 없습니다.", uid, cam_idx, frame_idx)
            if return_face:
                return None, None, None, None, None, False
            return None, None, None
        height, width = image.shape[:2]
        try:
            # 'L' 모드는 흑백(grayscale) 이미지로, 채널이 1개입니다.
            mask = np.array(Image.open(mask_b).convert('L'))  # H,W
        except:
            print(f"오류: 마스크 데이터가 없습니다.", uid, cam_idx, frame_idx)
            # 오류 발생 시, 검은색 마스크
            mask = np.zeros((height, width), dtype=np.float32)
        mask = (mask > 0.5).astype(np.float32)
        
        # face crop
        if return_face: 
            # --- (수정) ---
            # toc = self._toc_lru.get(uid) # <-- 이 줄 삭제 (인자로 받은 toc 사용)
            # --- (수정 끝) ---
            face_bboxes_b = toc.read_face_bbox_bytes(uid)
            face_bboxes = json.loads(face_bboxes_b) # 👈 **[수정]** .decode('utf-8') 제거
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
                #print(f"[jane] No face detect: {e}")
        
        
        # 2. crop image to enlarge human area.
        aspect_standard = 5.0 / 3  ## 532L in runners/infer/human_lrm.py
        #enlarge_ratio=[1.0, 1.0]  ## 532L in runners/infer/human_lrm.py
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
        
        # 3. resize to render_tgt_size for training
        tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
            cur_hw=image_patch.shape[:2],
            aspect_standard=aspect_standard,
            tgt_size=max_size,
            multiply=16,  ## 17L in runners/infer/human_lrm.py
        )  # (1696, 1024)

        image_patch_resized = cv2.resize(image_patch, (tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA)
        if intrinsic_matrix_rescale is not None: 
            intrinsic_matrix_rescale = scale_intrs(intrinsic_matrix_rescale, ratio_x=ratio_x, ratio_y=ratio_y)
            #assert (abs(intrinsic_matrix_rescale[:,0, 2] * 2 - image_patch_resized.shape[1]) < 2.5), f"{intrinsic_matrix_rescale[:,0, 2] * 2}, {image_patch_resized.shape[1]}"
            #assert (abs(intrinsic_matrix_rescale[:,1, 2] * 2 - image_patch_resized.shape[0]) < 2.5), f"{intrinsic_matrix_rescale[:,1, 2] * 2}, {image_patch_resized.shape[0]}"

            #intrinsic_matrix_rescale[:,0, 2] = image_patch_resized.shape[1] // 2
            #intrinsic_matrix_rescale[:,1, 2] = image_patch_resized.shape[0] // 2
        
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
            # 'L' 모드는 흑백(grayscale) 이미지로, 채널이 1개입니다.
            mask_pil = Image.open(mask_b).convert('L')
        except:
            print(f"오류: 마스크 데이터가 없습니다.", uid, cam_idx, frame_idx)
            # 오류 발생 시, 검은색 마스크를 대신 반환합니다.
            return torch.zeros(1, target_size[0], target_size[1])

        mask_pil = mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)

        transform = transforms.ToTensor()
        mask_tensor = transform(mask_pil)  # Shape: [1, H, W]
        return mask_tensor
    
    def load_smplx(self, uid: str, frame_idx: str): 
        key = (uid, frame_idx)
        # --- (수정) ---
        # 워커-로컬 프로퍼티 사용
        hit = self.smplx_lru.get(key)
        if hit is not None:
            return hit
        toc = self.toc_lru.get(uid)
        # --- (수정 끝) ---
        
        smplx_b, shape_param_b = toc.read_smplx_bytes(uid, frame_idx)
        smplx_data = None
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
            
            
            self.smplx_lru.put(key, smplx_data)
        except:
            print(f"Error: The smplx data was not found.", uid, frame_idx)
            smplx_data = None # 👈
            
        return smplx_data

    def load_smplx_history(self, uid: str, frame_idx: str, sampling_stride: int, min_frame_idx: int) -> np.ndarray: 
        history_smplx_poses = []
        
        current_frame_idx_int = int(frame_idx)

        for n in range(0, self.n_history_length+1):
            history_frame_idx_int = max(min_frame_idx, current_frame_idx_int - n * sampling_stride)
            history_frame_idx_str = f'{history_frame_idx_int:06d}'

            smplx_data = self.load_smplx(uid, history_frame_idx_str)
            
            if smplx_data is None: 
                # 👈 velocity 0이 되도록 제일 최근 pose로 stack. zero pose 말고.
                history_smplx_poses.append(history_smplx_poses[-1].copy())
            else: 
                # 👈 Dataset마다 좌표계 변환 행렬 상이
                rot_actorshq_to_dna = torch.tensor([[1.,0.,0.],
                                            [0.,-1.,0.],
                                            [0.,0.,-1.]], dtype=torch.float32)
                ## fullpose 
                pose_data = smplx_data['fullpose'].copy()
                root_pose_orig = axis_angle_to_matrix(torch.tensor(pose_data[0,:]))
                pose_data[0] = np.array(matrix_to_axis_angle(rot_actorshq_to_dna @ root_pose_orig))
                trans = np.array(rot_actorshq_to_dna @ torch.tensor(smplx_data['transl'])[:,None]).reshape(3)
                history_smplx_poses.append(np.vstack([pose_data, trans])) # (56, 3)

        smplx_pose_sequence = np.stack(history_smplx_poses, axis=0) # shape: (n_history_length + 1, 56, 3)
        return smplx_pose_sequence
        

    @no_proxy
    # 👈 inner_get_item 전체 교체
    def inner_get_item(self, idx):
        # 1. 샘플 정보 가져오기 (청크의 끝 프레임)
        uid, end_frame_idx_int = self.samples[idx]
        toc = self.toc_lru.get(uid) # 워커-로컬 toc
        F = self.num_train_frames

        # 2. 캐시된 데이터 가져오기
        ref_data = self.ref_data_cache[uid]
        cam_data = self.all_cam_params_cache[uid] # (1, V, 4, 4)

        # 3. 프레임 청크(F개) 로드
        min_frame_idx, _ = toc.get_min_max_frame_idx(uid, None)
        if min_frame_idx is None:
             raise RuntimeError(f"UID {uid} has no min/max frame index.")
        min_frame_idx_int = int(min_frame_idx)

        # 뷰 독립적인 데이터 (Frame-major list: [F] x ...)
        target_betas_list = []
        target_exprs_list = [] 
        target_pose_list = []
        target_transl_list = []
        motion_history_list = []
        frame_indices_list = [] 

        # Outer loop: Frames (F개)
        for i in range(self.num_train_frames):
            # 요청된 순서대로 프레임 인덱스 계산
            current_frame_idx_int = end_frame_idx_int - (self.num_train_frames - 1 - i) 
            current_frame_idx_str = f'{current_frame_idx_int:06d}'
            frame_indices_list.append(current_frame_idx_int)
            
            # --- Load View-Independent Data (Per-Frame) ---
            smplx_data = self.load_smplx(uid, current_frame_idx_str)
            
            num_betas = 10
            target_betas_list.append(torch.from_numpy(np.squeeze(smplx_data['betas'])[:num_betas]))
            target_exprs_list.append(torch.from_numpy(np.squeeze(smplx_data['expr'])[:num_betas]))
            
            
            target_pose_list.append(torch.from_numpy(smplx_data['fullpose']))
            target_transl_list.append(torch.from_numpy(smplx_data['transl']))
        
            skeleton_history = self.load_smplx_history(uid, current_frame_idx_str, self.sampling_stride, min_frame_idx_int)
            motion_history_list.append(torch.from_numpy(skeleton_history))
        
        # --- End of Frame Loop ---

        # --- 4. Final Stacking ---
        # 뷰 독립적인 데이터 스택 (dim=0) -> (F, ...)
        target_betas_tensor = torch.stack(target_betas_list, dim=0)
        target_exprs_tensor = torch.stack(target_exprs_list, dim=0) 
        
        target_poses_tensor = torch.stack(target_pose_list, dim=0)
        target_transls_tensor = torch.stack(target_transl_list, dim=0)
        motion_histories_tensor = torch.stack(motion_history_list, dim=0)
        frame_indices_tensor = torch.tensor(frame_indices_list, dtype=torch.int64)

        # --- 5. Expand Reference Data to Match Output Shape (F, 1, ...) ---
        ref_img_expanded = ref_data['img'].unsqueeze(0).unsqueeze(0).repeat(
            F, 1, 1, 1, 1
        )
        ref_face_img_expanded = ref_data['face_img'].unsqueeze(0).unsqueeze(0).repeat(
            F, 1, 1, 1, 1
        )
        ref_mask_expanded = ref_data['mask'].unsqueeze(0).unsqueeze(0).repeat(
            F, 1, 1, 1, 1
        )

        return {
            'uid': uid,
            'save_path': self.save_path,
            'frame_indices': frame_indices_tensor,      # (F,)
            
            # --- View-Dependent Target Data ---
            'intri': cam_data['intri'],                 # (1, V, 4, 4)
            'extri': cam_data['extri'],                 # (1, V, 4, 4)
            'height': cam_data['height'], # (1, V)
            'width': cam_data['width'], # (1, V)
            # 👈 (gt_img, gt_mask 제거됨)

            # --- View-Independent Target Data ---
            'target_betas': target_betas_tensor,    # (F, 10)
            'target_exprs': target_exprs_tensor, # 👈 꼭 빠뜨리지 말고 반환하기
            
            'target_pose': target_poses_tensor,     # (F, 55, 3)
            'target_transl': target_transls_tensor, # (F, 3)
            'motion_history': motion_histories_tensor, # (F, M, 56, 3)
            
            # --- Reference Data (Expanded) ---
            'img': ref_img_expanded,                # (F, 1, C, H, W)
            'face_img': ref_face_img_expanded,      # (F, 1, C, H, W)
            'mask': ref_mask_expanded,              # (F, 1, 1, H, W)
        }
    
    
    def close(self):
        try: 
            # 워커-로컬 객체가 생성되었을 경우에만 닫기 시도
            if self._toc_lru_local:
                self._toc_lru_local.close_all()
        except Exception: 
            pass

    def __del__(self): 
        self.close()