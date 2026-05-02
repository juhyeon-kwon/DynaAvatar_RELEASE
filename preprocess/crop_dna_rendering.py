import os
import cv2
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
cv2.setNumThreads(0)
import os.path as osp
import json
from tqdm import tqdm
from glob import glob
import argparse
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
import time


def load_cam_params(intri_path, extri_path, cam_id: str): 
    assert osp.exists(intri_path), f"cam param is not exists! {intri_path}"
    assert osp.exists(extri_path), f"cam param is not exists! {extri_path}"
    fs_intri = None
    fs_extri = None
    try: 
        fs_intri = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
        fs_extri = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
        
        K = fs_intri.getNode(f"K_{cam_id}").mat()  # (3x3)
        dist = fs_intri.getNode(f"dist_{cam_id}").mat().flatten()

        # --- Extrinsic Matrix ---  (w2c)
        Rvec = fs_extri.getNode(f'R_{cam_id}').mat()
        Tvec = fs_extri.getNode(f'T_{cam_id}').mat().flatten()
        if Rvec is not None:
            R_3x3, _ = cv2.Rodrigues(Rvec)
        else:
            R_3x3 = fs_extri.getNode(f'Rot_{cam_id}').mat()
        if R_3x3 is None or Tvec is None:
            raise ValueError(f"Rotation/Translation for '{cam_id}' not found.")
        
        T = Tvec.reshape(3)
        
    finally: 
        fs_intri.release()
        fs_extri.release()
        
    return K.astype(np.float32), dist.astype(np.float32).ravel(), R_3x3.astype(np.float32), T.astype(np.float32)

def process_cam_params(intri_path: str,
                       extri_path: str,
                       bbox_dict: dict,
                       cam_idx_list: List[str]): 
    cam_params_save_dict = {}
    for cam_idx in cam_idx_list: 
        K, dist, R, T = load_cam_params(intri_path, extri_path, cam_idx)
        x0, y0, _, _ = bbox_dict[cam_idx]['bbox']
        K[0,2] -= x0
        K[1,2] -= y0
        
        focal = np.array([K[0,0], K[1,1]], dtype=np.float32).tolist()
        princpt = np.array([K[0,2], K[1,2]], dtype=np.float32).tolist()
        cam_params_save_dict[cam_idx] = {'R': R.tolist(), 
                                         't': T.tolist(), 
                                         'focal': focal,
                                         'princpt': princpt, 
                                         'dist': np.array(dist, dtype=np.float32).tolist()}
    return cam_params_save_dict

def _crop_and_save_one(img_path, out_img_path, x0, y0, x1, y1):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: 
        print(f"[ERR] {img_path} is not exist!")
        return
    img_c = img[y0:y1+1, x0:x1+1, :]
    cv2.imwrite(out_img_path, img_c)
        
def crop_images(dataset_root_dir: str,
                uid: str,
                bbox_dict: dict,
                cam_idx_list: List[str], 
                out_dir: str,
                max_workers: int): 
    
    uid_root_dir = osp.join(dataset_root_dir, uid)
    images_root_dir = osp.join(uid_root_dir, 'images_orig')
    assert osp.exists(images_root_dir), f"{images_root_dir} is not exists!"
    
    for cam_idx in tqdm(cam_idx_list, desc=""): 
        img_path_list = glob(osp.join(images_root_dir, cam_idx, '*.jpg'))
        if len(img_path_list) == 0: 
            img_path_list = glob(osp.join(images_root_dir, cam_idx, '*.png'))
        
        x0, y0, x1, y1 = bbox_dict[cam_idx]['bbox']
        
        name = lambda p: osp.splitext(osp.basename(p))[0]
        img_map  = {name(p): p for p in img_path_list}

        img_out_dir  = osp.join(out_dir, 'images', cam_idx)
        os.makedirs(img_out_dir, exist_ok=True)
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for stem, ip in img_map.items():
                out_img_path  = osp.join(img_out_dir, f"{stem}.jpg")
                tasks.append(ex.submit(_crop_and_save_one, ip, out_img_path, x0, y0, x1, y1))
        for t in tasks:
            try:
                t.result()
            except Exception as e:
                print(f"[ERR] {e}")
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', type=str, dest='dataset_root_dir')
    parser.add_argument('--target_root_dir', type=str, dest='target_root_dir')
    parser.add_argument('--max_workers', type=int, default=8, dest='max_workers')
    args = parser.parse_args()
    assert args.dataset_root_dir, "Please set datset_root_dir."
    assert args.target_root_dir, "Please set target_root_dir."
    return args


args = parse_args()
dataset_root_dir = args.dataset_root_dir
target_root_dir = args.target_root_dir
max_workers = args.max_workers

uid_list = os.listdir(dataset_root_dir)
if 'DNA' in dataset_root_dir: 
    uid_list.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))

print("Target uid list length:", len(uid_list))

for uid in tqdm(uid_list): 
    print(f"[START] {uid}")
    images_root_dir = osp.join(dataset_root_dir, uid, 'images_orig')
    cam_idx_list = os.listdir(images_root_dir)
    cam_idx_list.sort(key=lambda x: int(x))
    
    with open(osp.join(target_root_dir, uid, 'bbox_orig.json')) as f: 
        bbox_orig_dict = json.load(f)
    
    if not osp.exists(osp.join(target_root_dir, uid, 'cam_params.json')): 
        intri_path = osp.join(dataset_root_dir, uid, "intri.yml")
        extri_path = osp.join(dataset_root_dir, uid, "extri.yml")
        cam_params_save_dict = process_cam_params(intri_path=intri_path, extri_path=extri_path, bbox_dict=bbox_orig_dict, cam_idx_list=cam_idx_list)
        
        with open(osp.join(target_root_dir, uid, 'cam_params.json'), 'w') as f: 
            json.dump(cam_params_save_dict, f, indent=4) 
    
    with open(osp.join(target_root_dir, uid, 'cam_params.json')) as f: 
        cam_params_dict = json.load(f)
    
    if len(cam_params_dict.keys()) != len(bbox_orig_dict.keys()): 
        print(f"[ERR] {uid} len(cam_params.json) != len(bbox_orig.json)", len(cam_params_dict.keys()), len(bbox_orig_dict.keys()))            
    
    start = time.time()
    crop_images(
        dataset_root_dir=dataset_root_dir,
        uid=uid,
        bbox_dict=bbox_orig_dict,
        cam_idx_list=cam_idx_list, 
        out_dir=osp.join(target_root_dir, uid),
        max_workers=max_workers
    )
    print(f"[FIN] {uid}:", round(time.time() - start, 2))
    print("=================================================")
    
   