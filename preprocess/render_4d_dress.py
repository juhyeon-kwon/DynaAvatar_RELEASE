import argparse
from tqdm import tqdm
import os
import os.path as osp
import torch
import numpy as np
import cv2
from PIL import Image
import json
import glob
import pickle
import trimesh

from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer import (
    PointLights,
    HardPhongShader,
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
)


def arg_parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', type=str, required=True, dest='dataset_root_dir', help='root directory containing original dataset.')
    parser.add_argument('--target_root_dir', type=str, required=True, dest='target_root_dir', help='root directory containing reannotated dataset.')
    parser.add_argument('--batch_size', type=int, default=2, dest='batch_size', help='render 24 * batch_size frames per iteration.')
    args = parser.parse_args()
    return args


class PytorchRenderer:
    """
    Pytorch3d Multi-View Image Renderer.
    """

    def __init__(self, num_view=24, img_size=512, max_x=1.0, min_x=-1.0, max_y=1.0, min_y=-1.0,
                 init_model=False, dtype=torch.float32, device=torch.device('cuda:0')):

        # init dtype, device
        self.dtype = dtype
        self.device = device
        # init num_view, img_size
        self.num_view = num_view
        self.img_size = img_size
        self.max_x, self.min_x, self.max_y, self.min_y = max_x, min_x, max_y, min_y

        # not init_model
        if not init_model: return

        # init front view with num_view = 1
        if self.num_view == 1:
            R, T = look_at_view_transform(dist=5.0, elev=0, azim=0, up=((0, 1, 0),), at=((0, 0, 0),))
        # init horizontal views with num_view = 6, 12
        elif self.num_view == 6 or self.num_view == 12:
            azim = torch.linspace(0, 360, self.num_view + 1)[:self.num_view]
            R, T = look_at_view_transform(dist=5.0, elev=0, azim=azim, up=((0, 1, 0),), at=((0, 0, 0),))
        # init horizontal, upper, and lower views with num_view = 24
        elif self.num_view == 24:
            # init horizontal views
            azim = torch.linspace(0, 360, self.num_view // 2 + 1)[:self.num_view // 2]
            R_normal, T_normal = look_at_view_transform(dist=5.0, elev=0, azim=azim, up=((0, 1, 0),), at=((0, 0, 0),))
            # init upper views
            azim = torch.linspace(0, 360, self.num_view // 4 + 1)[:self.num_view // 4]
            R_upper, T_upper = look_at_view_transform(dist=5.0, elev=30, azim=azim, up=((0, 1, 0),), at=((0, 0, 0),))
            # init lower views
            azim = torch.linspace(0, 360, self.num_view // 4 + 1)[:self.num_view // 4]
            R_lower, T_lower = look_at_view_transform(dist=5.0, elev=-30, azim=azim, up=((0, 1, 0),), at=((0, 0, 0),))
            # cat final views
            R = torch.cat([R_normal, R_upper, R_lower], dim=0)
            T = torch.cat([T_normal, T_upper, T_lower], dim=0)
        else:
            raise Exception('Invalid view number: {} over {6, 12, 24}'.format(self.num_view))
        
        wx = self.max_x - self.min_x
        wy = self.max_y - self.min_y
        
        Zref = 5.0
        fx = img_size / wx * Zref
        fy = img_size / wy * Zref
        cx = img_size / 2
        cy = img_size / 2
        
        self.R = R.to(device) # (num_view,3,3)
        self.T = T.to(device)  # (num_view,3)
        self.focal = torch.tensor([fx, fy], dtype=torch.float32, device=device)
        self.princpt = torch.tensor([cx, cy], dtype=torch.float32, device=device)
        
        dummy_cams = PerspectiveCameras(focal_length=((fx,fy),),
                                        principal_point=((cx,cy),),
                                        image_size=((img_size,img_size),),
                                        R=self.R[:1], T=self.T[:1],
                                        in_ndc=False, device=device)

        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]], ambient_color=((1, 1, 1),), diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),))
        self.shader = HardPhongShader(device=self.device, cameras=dummy_cams, lights=self.lights)
        self.mesh_raster_settings = RasterizationSettings(image_size=self.img_size, faces_per_pixel=2, bin_size=128, blur_radius=0, cull_backfaces=False, max_faces_per_bin=30000)
        self.mesh_rasterizer = MeshRasterizer(cameras=dummy_cams, raster_settings=self.mesh_raster_settings)
        self.mesh_renderer = MeshRenderer(rasterizer=self.mesh_rasterizer, shader=self.shader)

    # render scan_mesh to multi_view color images
    @torch.no_grad()
    def render_mesh_images(self, mesh_list: list[Meshes]):
        # init render_images
        render_images = dict()
        batch_size = len(mesh_list)
        
        R = self.R.repeat(batch_size,1,1)  # (num_view*batch_size,3,3)
        T = self.T.repeat(batch_size,1)     # (num_view*batch_size
        
        cameras = PerspectiveCameras(
            focal_length=self.focal[None].expand(batch_size * self.num_view, -1),
            principal_point=self.princpt[None].expand(batch_size * self.num_view, -1),
            image_size=torch.tensor([[self.img_size, self.img_size]], device=self.device).expand(batch_size * self.num_view, -1),
            R=R, T=T, in_ndc=False, device=self.device
        )
        
        meshes = join_meshes_as_batch(mesh_list)
        # render color and mask image with num_view
        image_color = self.mesh_renderer(meshes.extend(self.num_view), cameras=cameras, raster_settings=self.mesh_raster_settings, lights=self.lights)
        render_images['color'] = image_color[..., :3]  # (batch_size*num_view, h, w, 3), in [0, 1]
        render_images['mask'] = image_color[..., -1]  # (batch_size*num_view, h, w)
        return render_images


class DatasetUtils: 
    def __init__(self, dataset_dir='', save_root_dir='', preprocess_init=False, dtype=torch.float32, device=torch.device('cuda:0'), log_idx='0'):
        self.dtype = dtype
        self.device = device
        self.dataset_dir = dataset_dir
        self.preprocess_init = preprocess_init
        self.save_root_dir = save_root_dir
    
    def preprocess_scan_mesh(self, mesh, mcentral=False, bbox=True, rotation=None, offset=None, scale=1.0):
        mcenter = np.mean(mesh['vertices'], axis=0)
        bmax = np.max(mesh['vertices'], axis=0)
        bmin = np.min(mesh['vertices'], axis=0)
        bcenter = (bmax + bmin) / 2
        if mcentral:
            mesh['vertices'] -= mcenter
        elif bbox:
            mesh['vertices'] -= bcenter
        
        mesh['vertices'] /= scale
        if rotation is not None:
            mesh['vertices'] = np.matmul(rotation, mesh['vertices'].T).T
        if offset is not None:
            mesh['vertices'] += offset
        
        return mesh, {'mcenter': mcenter, 'bcenter': bcenter}, scale, bmax, bmin
    
    def load_scan_mesh(self, mesh_fn):
        atlas_fn = mesh_fn.replace('mesh-', 'atlas-')
        mesh_data = pickle.load(open(mesh_fn, "rb"))
        atlas_data = pickle.load(open(atlas_fn, "rb"))
        uv_image = Image.fromarray(atlas_data).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
        
        scan_mesh = {
            'vertices': mesh_data['vertices'].copy(),
            'faces': mesh_data['faces'],
            'uvs': mesh_data['uvs'],
            'uv_image': np.array(uv_image),
        }
        
        # preprocess scan mesh: scale, centralize, normalize, rotation, offset
        scan_mesh, _, _, bmax, bmin = self.preprocess_scan_mesh(scan_mesh, mcentral=False, bbox=False, scale=1.0)
        
        return scan_mesh, bmax, bmin
    
    def preprocess(self, subject_id, outfit, seq, batch_size): 
        save_dir = osp.join(self.save_root_dir, f"{subject_id}_{outfit.lower()}_{seq[4:]}") 
        os.makedirs(save_dir, exist_ok=True)
        
        hyper_params = dict()
        seq_root_dir = os.path.join(self.dataset_dir, subject_id, outfit, seq)
        scan_dir = os.path.join(seq_root_dir, 'Meshes_pkl')
        scan_file_list = sorted(glob.glob(os.path.join(scan_dir, 'mesh-f*.pkl')))
        scan_frames_list = [fn.split('/')[-1].split('.')[0][-5:] for fn in scan_file_list]
        scan_nums, scan_frames = len(scan_frames_list), scan_frames_list
        
        num_view = 24
        if osp.exists(osp.join(save_dir, 'images', f'{num_view-1:02d}')) and len(os.listdir(osp.join(save_dir, 'images', f'{num_view-1:02d}'))) == scan_nums: 
            print(f"[SKIP] {subject_id}_{outfit.lower()}_{seq[4:]} exists")
            return
        
        scan_meshes = []
        seq_bmax, seq_bmin = None, None
        for n_frame in tqdm(range(scan_nums)):
            frame = scan_frames[n_frame]
            scan_mesh_fn = os.path.join(scan_dir, 'mesh-f{}.pkl'.format(frame))
            scan_mesh, bmax, bmin = self.load_scan_mesh(scan_mesh_fn)
            scan_meshes.append(scan_mesh)
            if seq_bmax is None: 
                seq_bmax, seq_bmin = bmax, bmin
            else: 
                seq_bmax = np.maximum(seq_bmax, bmax)
                seq_bmin = np.minimum(seq_bmin, bmin)
        seq_center = (seq_bmax + seq_bmin) / 2
        size_x, size_y, size_z = seq_bmax - seq_bmin
        size = max(size_x, size_y, size_z)
        size = size * 1.2
                
        render = PytorchRenderer(num_view=num_view, img_size=1920, init_model=True, dtype=self.dtype, device=self.device, min_x=-size/2, max_x=size/2, min_y=-size/2, max_y=size/2)
        
        # Save cam_params
        cam_idx_list = [f"{i:02d}" for i in range(render.num_view)]
        cam_params = dict()
        S = torch.diag(torch.FloatTensor([-1, -1, 1]))
        for i, cam_idx in enumerate(cam_idx_list): 
            os.makedirs(osp.join(save_dir, 'images', cam_idx), exist_ok=True)
            os.makedirs(osp.join(save_dir, 'masks', cam_idx), exist_ok=True)
            R = S @ render.R[i].cpu().T
            cam_params[cam_idx] = {
                'R': R.tolist(),
                't': render.T[i].cpu().tolist(),
                'focal': render.focal.cpu().tolist(),
                'princpt': render.princpt.cpu().tolist(),
                'dist': []
            }
        with open(osp.join(save_dir, 'cam_params.json'), 'w') as f: 
            json.dump(cam_params, f, indent=4)
        
        mesh_list = []
        frame_list = []
        # Render scan & mask images for each frame and view
        for n_frame in tqdm(range(scan_nums), desc=f"{subject_id}_{outfit.lower()}_{seq[4:]}"): 
            frame = scan_frames[n_frame]
            scan_mesh = scan_meshes[n_frame]
            th_verts = torch.tensor(scan_mesh['vertices']-seq_center, dtype=self.dtype, device=self.device)
            th_faces = torch.tensor(scan_mesh['faces'], dtype=torch.long, device=self.device)
            th_uvs = torch.tensor(scan_mesh['uvs'], dtype=self.dtype, device=self.device)
            th_uv_image = torch.tensor(cv2.resize(scan_mesh['uv_image'], (1024, 1024)) / 255., dtype=self.dtype, device=self.device).unsqueeze(0)
            textures = TexturesUV(
                maps=th_uv_image,                 # [1,H,W,3]
                faces_uvs=[th_faces],           # list of LongTensor [F,3]
                verts_uvs=[th_uvs],           # list of FloatTensor [V,2]
            )
            
            mesh_list.append(Meshes(verts=[th_verts], faces=[th_faces], textures=textures))
            frame_list.append(frame)
            
            if len(mesh_list) < batch_size and n_frame < scan_nums - 1: 
                continue
            
            render_results = render.render_mesh_images(mesh_list=mesh_list)
            render_images = (render_results['color'].cpu().numpy() * 255.).astype(np.uint8)  # (B*nv, h, w, 3)
            render_masks = ((render_results['mask'].cpu().numpy() > 0.) * 255.).astype(np.uint8)
            
            for i, (render_image, render_mask) in enumerate(zip(render_images, render_masks)): 
                cam_idx = f"{i % num_view:02d}"
                frame_idx = i // num_view
                frame = frame_list[frame_idx]
                save_img_dir = osp.join(save_dir, 'images', cam_idx)
                save_mask_dir = osp.join(save_dir, 'masks', cam_idx)
                self._save_one(render_image, render_mask, save_img_dir, save_mask_dir, frame)
            
            mesh_list = []
            frame_list = []
    
    def _save_one(self, img_rgb, mask_u8, save_img_dir, save_mask_dir, frame_id):
        img_bgr = np.ascontiguousarray(img_rgb[..., ::-1])
        msk = np.ascontiguousarray(mask_u8)

        img_path = osp.join(save_img_dir,  f"{int(frame_id):06d}.jpg")
        msk_path = osp.join(save_mask_dir, f"{int(frame_id):06d}.png")
        cv2.imwrite(img_path, img_bgr)
        if msk.ndim == 3: 
            msk = msk[..., 0]
        cv2.imwrite(msk_path, msk)


args = arg_parser()
dataset_root_dir = args.dataset_root_dir
target_root_dir = args.target_root_dir
batch_size = args.batch_size

subject_id_list = os.listdir(dataset_root_dir)
subject_id_list.sort(key=lambda x: int(x))

print("Target subject id len:", len(subject_id_list))
preprocessor = DatasetUtils(dataset_dir=dataset_root_dir, save_root_dir=target_root_dir)
    
for subject_id in subject_id_list: 
    outfit_list = os.listdir(osp.join(dataset_root_dir, subject_id))
    for outfit in outfit_list: 
        seq_list = os.listdir(osp.join(dataset_root_dir, subject_id, outfit))
        seq_list = [seq for seq in seq_list if 'Take' in seq]
        seq_list.sort(key=lambda x: int(x[4:]))
        for seq in seq_list:            
            preprocessor.preprocess(subject_id=subject_id, outfit=outfit, seq=seq, batch_size=batch_size)