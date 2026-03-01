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

########## debugging
'''import smplx
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, TexturesVertex, PointLights, Materials, SoftPhongShader
MODEL_DIR_TMP = '/data/human_model_files'
def tnp(x): return torch.from_numpy(x).float()

model_tmp = smplx.create(
        model_path=MODEL_DIR_TMP, model_type="smplx", gender='neutral',
        use_pca=False, use_face_contour=True, num_betas=10, num_expression_coeffs=10, batch_size=1
    )

def mesh_render(mesh, face, cam_param, render_shape):         
    render_height, render_width = render_shape

    # get visible faces from mesh
    fragments, imgs = get_face_index_map_xy(mesh, face, cam_param, (render_height, render_width))
    # imgs: (1,H,W,4) RGBA
    
    # fg mask
    pix_to_face = fragments.pix_to_face # batch_size, render_height, render_width, faces_per_pixel. invalid: -1
    mask = pix_to_face[0, ..., 0] >= 0  # 메쉬가 그려진 픽셀
    
    return imgs[0,:,:,:3], mask
    
def get_face_index_map_xy(mesh, face, cam_param, render_shape): 
    lights = PointLights(location=[[0.0, 0.0, 0.0]], ambient_color=((0.4,0.4,0.4),), diffuse_color=((0.6,0.6,0.6),), specular_color=((0.0,0.0,0.0),))
    materials = Materials()
    
    batch_size = mesh.shape[0]
    face = torch.from_numpy(face)[None,:,:].expand(batch_size,-1,-1)
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    
    base_color = torch.tensor([[0.8, 0.8, 0.8]])
    verts_rgb = base_color.expand(mesh.shape[1], -1).unsqueeze(0)          # (1,V,3)
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(mesh, face, textures=textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                principal_point=cam_param['princpt'],
                                device='cpu',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).view(1,2))
    
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    shader = SoftPhongShader(cameras=cameras, lights=lights, materials=materials)
    images = shader(fragments, mesh)
    
    return fragments, images

def render_mesh(beta, fullpose, trans, bg, intrinsic, extrinsic): 
    # smplx
    # fullpose 55,3
    root_pose = fullpose[0].reshape(1,3)
    body_pose = fullpose[1:22].reshape(1,-1)
    lhand_pose = fullpose[22:37].reshape(1,-1)
    rhand_pose = fullpose[37:52].reshape(1,-1)
    jaw_pose = fullpose[52].reshape(1,3)
    leye_pose = fullpose[53].reshape(1,3)
    reye_pose = fullpose[54].reshape(1,3)
    expr = torch.zeros((1,10))
    betas = beta.reshape(1,-1)
    transl = trans.reshape(1,3)
    output = model_tmp(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=betas, transl=transl)
        
    # camera-centered 3D coordinate
    verts = output.vertices[0].cpu().numpy()  # (N,3), meters
    faces = model_tmp.faces.astype(np.int64)
    R = extrinsic[:3,:3]  # c2w
    R_w2c = R.T
    t = extrinsic[:3,3]
    t_w2c = -(R.T @ t.reshape(3,1)).reshape(3,)
    mesh_cam = verts @ R_w2c.T + t_w2c.reshape(1,3) # v,3
    
    
    img_height, img_width = bg.shape[1:]
    cam_param = {
        'focal': np.array([intrinsic[0,0], intrinsic[1,1]]), 
        'princpt': np.array([intrinsic[0,2], intrinsic[1,2]]), 
        'R': R_w2c, 
        't': t_w2c,
    }
    render_img, mask = mesh_render(torch.from_numpy(mesh_cam[None,:,:]), faces, {k: v[None] for k,v in cam_param.items()}, render_shape=(img_height, img_width))
    return render_img, mask'''
#############

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
    
    '''# image center
    cx = (W / 2.0)
    cy = (H / 2.0)
    
    half_w = max(abs(cx - x_min), abs(x_max - cx))
    half_h = max(abs(cy - y_min), abs(y_max - cy))  # [x0:x1] 에서 x1 은 포함되지 않으므로
    half_w = max(half_w, round(half_h / aspect_standard))
    need_w = int(2 * half_w)
    need_h = round(need_w * aspect_standard)
    
    x0 = math.floor(cx - half_w)
    y0 = math.floor(cy - need_h / 2.0)
    x1 = x0 + need_w
    y1 = y0 + need_h'''

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
 
__all__ = ['DNARenderingDataset']

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

class DNARenderingDataset(BaseDataset):

    def __init__(self, root_dirs: list[str], meta_path: str,
                 use_flame: bool,
                 src_head_size: int,
                 n_history_length : int, 
                 fps : int,
                 smplx_lru_size: int = 256,   # jane;
                 toc_lru_capacity: int = 64,   # jane;
                 sqlite_mmap_size: int = 128 << 20,   # jane;
                 **data_kwargs
                 ): # qw00n; arguments from configs : **dataset_kwargs
        super().__init__(root_dirs, meta_path)

        self.n_history_length = n_history_length
        self.num_train_views = 1 # qw00n; TODO
        self.num_train_frames = 4 # qw00n; TODO
        self.fps = fps
        '''
        {'0095_01': ['motion_simple'], '0047_12': ['motion_simple'], '0307_07': ['motion_simple'], '0102_02': ['interaction_hard'], '0123_02': ['interaction_hard'], '0174_09': ['interaction_hard'], '0022_10': ['motion_hard'], '0113_06': ['motion_hard'], '0018_05': ['motion_hard'], '0019_06': ['texture_simple'], '0094_02': ['texture_simple'], '0241_10': ['texture_simple'], '0124_03': ['motion_medium'], '0166_04': ['motion_medium'], '0111_08': ['motion_medium'], '0206_04': ['deformation_hard'], '0008_01': ['deformation_hard'], '0121_02': ['deformation_hard'], '0047_01': ['texture_hard'], '0097_04': ['texture_hard'], '0188_02': ['texture_hard'], '0025_11': ['deformation_medium'], '0012_09': ['deformation_medium'], '0115_07': ['deformation_medium'], '0152_01': ['interaction_no'], '0235_11': ['interaction_no'], '0307_03': ['interaction_no'], '0034_04': ['deformation_simple'], '0031_03': ['deformation_simple'], '0310_04': ['deformation_simple'], '0239_01': ['interaction_medium'], '0128_04': ['interaction_medium'], '0133_07': ['interaction_medium'], '0196_09': ['interaction_simple'], '0118_07': ['interaction_simple'], '0309_03': ['interaction_simple'], '0219_07': ['texture_medium'], '0165_08': ['texture_medium'], '0147_04': ['texture_medium']}
        '''       

        # jane;
        self._toc_lru = TocLRU(self.root_dirs,
                               capacity=toc_lru_capacity,
                               sqlite_mmap_size=sqlite_mmap_size)  # worker-local: uid -> TarFile
        self._cam_params_cache = {}          # uid -> (K_4x4, c2w_4x4)
        self._smplx_lru = LRU(capacity=smplx_lru_size)         # (uid, frame) -> npz dict (LRU)
        ##########
        

    # def other methods if needed   

    # jane;
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
            image = np.array(Image.open(img_b).convert('RGB'))  # H,W,C
            image = (image / 255.).astype(np.float32)
        except: 
            print(f"오류: 이미지 데이터가 없습니다.", uid, cam_idx, frame_idx)
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
                #print(f"[jane] No face detect: {e}")
        
        '''# 1. "resize" image keeping astpect ratio: memory, 속도 절감 위해 우선 resize (downsample)
        rgb = resize_image_keepaspect_np(image, max_tgt_size=896)
        mask = resize_image_keepaspect_np(mask, max_tgt_size=896)
        if intrinsic_matrix is not None: 
            ratio = rgb.shape[0] / height
            intrinsic_matrix = scale_intrs(intrinsic_matrix, ratio_x=ratio, ratio_y=ratio)  # (1,4,4)'''
        
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
        hit = self._smplx_lru.get(key)
        if hit is not None:
            return hit
        toc = self._toc_lru.get(uid)
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
                
            self._smplx_lru.put(key, smplx_data)
        except:
            print(f"Error: The smplx data was not found.", uid, frame_idx)
            
        return smplx_data

    def load_smplx_history(self, uid: str, frame_idx: str, sampling_stride: int, min_frame_idx: int) -> np.ndarray: 
        history_smplx_poses = []
        
        current_frame_idx_int = int(frame_idx)
        for n in range(0, self.n_history_length+1):
            history_frame_idx_int = max(min_frame_idx, current_frame_idx_int - n * sampling_stride)
            history_frame_idx_str = f'{history_frame_idx_int:06d}'
            smplx_data = self.load_smplx(uid, history_frame_idx_str)
            
            if smplx_data is None: 
                zero_pose = np.zeros((56, 3), dtype=np.float32)
                history_smplx_poses.append(zero_pose)
            else: 
                ## fullpose
                pose_data = smplx_data['fullpose']
                history_smplx_poses.append(np.vstack([pose_data, smplx_data['transl']])) # (56, 3)

        smplx_pose_sequence = np.stack(history_smplx_poses, axis=0) # shape: (n_history_length + 1, 56, 3)
        return smplx_pose_sequence
        

    @no_proxy
    def inner_get_item(self, idx):
        uid = list(self.uids)[idx]
        subject_path = os.path.join(self.root_dirs, uid) # /data/DNA_Rendering/DNA_Rendering/0018_05
        
        # jane;
        toc = self._toc_lru.get(uid)
        
        # jane; sample ref_cam_idx from face_bboxes cam_idx list
        face_bboxes_b = toc.read_face_bbox_bytes(uid)
        face_bboxes = json.loads(face_bboxes_b)
        face_bboxes_cam_idx_list = tuple(face_bboxes.keys())  # TODO: caching?
        if not face_bboxes_cam_idx_list:
            raise RuntimeError(f"{uid}: face_bboxes is empty")
        ref_cam_idx = random.choice(face_bboxes_cam_idx_list)
        
        # jane; get min frame idx and use as ref_frame
        min_frame_idx, max_frame_idx = toc.get_min_max_frame_idx(uid, ref_cam_idx)
        #ref_frame_idx = '000006'
        ref_frame_idx = f"{int(min_frame_idx)+6:06d}"

        random_idx = random.randint(13, 34) # qw00n; near-front view
        tar_cam_idx = f'{random_idx:02d}'

        # Load cam param
        intrinsic_matrix, extrinsic_matrix = self.load_camera_params(uid, tar_cam_idx)  # (1,4,4), (1,4,4)

        
        #target_face_bboxes_b = toc.read_face_bbox_targ_bytes(uid)
        #target_face_bboxes = json.loads(target_face_bboxes_b)
        #print(target_face_bboxes)
        #exit()
        
        # Load & resize ref img, mask, and cam_param
        ref_img_b = toc.read_image_bytes(uid, ref_cam_idx, ref_frame_idx)
        ref_mask_b = toc.read_mask_bytes(uid, ref_cam_idx, ref_frame_idx)
        ref_img, ref_mask, ref_cam_param, ref_face_img, _, is_ref_face_detected = self.load_and_resize_image_to_tensor(uid, ref_cam_idx, ref_frame_idx, io.BytesIO(ref_img_b), io.BytesIO(ref_mask_b), intrinsic_matrix, return_face=True) # 3, 1024, 856 (C, H, W) #### TODO
        
        #cv2.imwrite(os.path.join('/home/jane/workspace/moLHM/temp', f'{uid}_{tar_cam_idx}_face_img.jpg'), ref_face_img.clone().detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255.)
        
        sampling_stride = max(self.fps // 15, 1)
        assert sampling_stride == 1
        
        #start_sampling_idx = 10
        start_sampling_idx = int(min_frame_idx) + sampling_stride * (self.n_history_length+self.num_train_frames)  # jane; TODO: fix hard-coding
        end_frame_idx_int = random.randint(min(start_sampling_idx, int(max_frame_idx)), int(max_frame_idx))

        gt_img_list, gt_mask_list = [], []
        target_betas_list, target_pose_list, target_transl_list = [], [], []
        gt_cam_param_list = []
        motion_history_list = []
        target_face_bbox_list, is_target_face_detected_list = [], []
        
        for i in range(self.num_train_frames): 
            current_frame_idx_int = end_frame_idx_int - (self.num_train_frames-1-i) * sampling_stride
            current_frame_idx_str = f'{current_frame_idx_int:06d}'
            
            gt_img_b = toc.read_image_bytes(uid, tar_cam_idx, current_frame_idx_str)
            gt_mask_b = toc.read_mask_bytes(uid, tar_cam_idx, current_frame_idx_str)
            gt_img, gt_mask, gt_cam_param, _, target_face_bbox, is_target_face_detected  = self.load_and_resize_image_to_tensor(uid, tar_cam_idx, current_frame_idx_str, io.BytesIO(gt_img_b), io.BytesIO(gt_mask_b), intrinsic_matrix, return_face=True) # 3, 1024, 856 (C, H, W) #### TODO
            
            gt_img_list.append(gt_img)
            gt_mask_list.append(gt_mask)
            gt_cam_param_list.append(gt_cam_param)
            
            smplx_data = self.load_smplx(uid, current_frame_idx_str)
            
            num_betas = 10
            target_betas_list.append(torch.from_numpy(np.squeeze(smplx_data['betas'])[:num_betas]))
            target_pose_list.append(torch.from_numpy(smplx_data['fullpose']))
            target_transl_list.append(torch.from_numpy(smplx_data['transl']))
            
            skeleton_history = self.load_smplx_history(uid, current_frame_idx_str, sampling_stride, int(min_frame_idx))
            motion_history_list.append(torch.from_numpy(skeleton_history))
            
            target_face_bbox_list.append(target_face_bbox)
            is_target_face_detected_list.append(is_target_face_detected)
        
        gt_imgs_tensor = torch.stack(gt_img_list, dim=0)
        gt_masks_tensor = torch.stack(gt_mask_list, dim=0)
        target_betas_tensor = torch.stack(target_betas_list, dim=0)
        target_poses_tensor = torch.stack(target_pose_list, dim=0)
        target_transls_tensor = torch.stack(target_transl_list, dim=0)
        motion_histories_tensor = torch.stack(motion_history_list, dim=0)
        target_face_bboxes_tensor = torch.stack(target_face_bbox_list, dim=0)
        is_target_face_detected_tensor = torch.tensor(is_target_face_detected_list)
        
        
        '''for i in range(self.num_train_frames): 
            current_frame_idx_int = end_frame_idx_int - (self.num_train_frames-1) + i
            current_frame_idx_str = f'{current_frame_idx_int:06d}'
            cv2.imwrite(os.path.join('/home/jane/workspace/moLHM/temp', f'{uid}_{tar_cam_idx}_{current_frame_idx_str}_img.jpg'), gt_imgs_tensor[i,:,:,:].clone().detach().cpu().numpy().transpose(1,2,0)*255.)
            cv2.imwrite(os.path.join('/home/jane/workspace/moLHM/temp', f'{uid}_{tar_cam_idx}_{current_frame_idx_str}_mask.jpg'), gt_masks_tensor[i,:,:,:].clone().detach().cpu().numpy().transpose(1,2,0)*255.)'''
        #cv2.imwrite(os.path.join('/home/jane/workspace/moLHM/temp', f'{uid}_{tar_cam_idx}_img.jpg'), gt_imgs_tensor[0,:,:,:].clone().detach().cpu().numpy().transpose(1,2,0)*255.)
        #cv2.imwrite(os.path.join('/home/jane/workspace/moLHM/temp', f'{uid}_{tar_cam_idx}_mask.jpg'), gt_masks_tensor[0,:,:,:].clone().detach().cpu().numpy().transpose(1,2,0)*255.)
        
        
        #print(f"[{uid}] fx, fy, cx, cy at dataloader:", gt_cam_param_list[0][0,0,0], gt_cam_param_list[0][0,1,1], gt_cam_param_list[0][0,0,2], gt_cam_param_list[0][0,1,2])
        #print(f'[{uid}] trans at dataloader:', target_transls_tensor[0])
        ##### smplx rendering test
        '''render_img, mask = render_mesh(target_betas_tensor[0], target_poses_tensor[0], target_transls_tensor[0], gt_imgs_tensor[0], gt_cam_param_list[0][0], extrinsic_matrix[0])
        
        render_img = (render_img.cpu().numpy() * 255).astype(np.uint8)
        render_img = render_img[...,::-1]
        
        gt_img = gt_imgs_tensor[0].clone().permute(1,2,0).numpy()[:,:,::-1]*255
        
        #print('shape:', render_img.shape, gt_img.shape)
        gt_img[mask] = render_img[mask]  # h, w, 3

        cv2.imwrite(os.path.join('/home/jane/workspace/moLHM/temp', f"{uid}_{tar_cam_idx}_smplx.jpg"), gt_img)
        '''
        #########################
        
        
        return {
            'uid': uid,
            'intri': np.concatenate(gt_cam_param_list, axis=0),
            'extri': np.repeat(extrinsic_matrix, self.num_train_frames, axis=0),
            'img': ref_img.unsqueeze(0).repeat(self.num_train_frames, 1, 1, 1),     
            'face_img': ref_face_img.unsqueeze(0).repeat(self.num_train_frames, 1, 1, 1),            
            'is_ref_face_detected': torch.tensor([is_ref_face_detected]).repeat(self.num_train_frames),     
            'mask': ref_mask.unsqueeze(0).repeat(self.num_train_frames, 1, 1, 1),               
            
            # --- Stacked Target Data ---
            'gt_img': gt_imgs_tensor,               # shape: (num_train_frames, C, H, W)
            'gt_mask': gt_masks_tensor,             # shape: (num_train_frames, 1, H, W)
            'target_betas': target_betas_tensor,    # shape: (num_train_frames, 10)
            'target_pose': target_poses_tensor,     # shape: (num_train_frames, 55, 3)
            'target_transl': target_transls_tensor, # shape: (num_train_frames, 3)
            'motion_history': motion_histories_tensor, # shape: (num_train_frames, M, 56, 3)
            'is_target_face_detected': is_target_face_detected_tensor,  # shape: (num_train_frames,)
            'target_face_bboxes_tensor': target_face_bboxes_tensor  # shape: (num_train_frames, 4)
        }
    
    
    def close(self):
        try: 
            self._toc_lru.close_all()
        except Exception: 
            pass

    def __del__(self): 
        self.close()