import argparse
import os
import os.path as osp
import cv2
from glob import glob
import json
import torch
import numpy as np

def change_kpt_name(src_kpt, src_name, dst_name):
    src_kpt_num = len(src_name)
    dst_kpt_num = len(dst_name)

    new_kpt = np.zeros(((dst_kpt_num,) + src_kpt.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_kpt[dst_idx] = src_kpt[src_idx]

    return new_kpt

def make_video(video_output_path, output_path):
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(output_path, '*.png'))])
    img_height, img_width = cv2.imread(osp.join(output_path, str(frame_idx_list[0]) + '.png')).shape[:2]
    video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width, img_height))
    for frame_idx in frame_idx_list:
        frame = cv2.imread(osp.join(output_path, str(frame_idx) + '.png'))
        frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
        video.write(frame.astype(np.uint8))
    video.release()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--view_num', type=str, dest='view_num')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    assert args.root_path, "Please set view_num."
    return args

args = parse_args()
root_path = args.root_path
view_num = args.view_num

MODEL_NAME = 'LHM-500M-HF'
IMAGE_PATH_OR_FOLDER = osp.join(root_path, 'images')
MOTION_SEQ = './train_data/motion_video/neutral_rotation/smplx_params'
RETURN_GS = True
VIEW_NUM = view_num

## Make canonical(大) smplx pose
cmd = f"rm {osp.join(MOTION_SEQ, '*')}"
os.system(cmd)
cmd = f'python get_neutral_pose_rotation.py --save_path {MOTION_SEQ} --view_num {VIEW_NUM}'
os.system(cmd)

## Get initial 3DGS & canonical posed views
cmd = f'bash ./inference.sh {MODEL_NAME} {IMAGE_PATH_OR_FOLDER} {MOTION_SEQ} {RETURN_GS}'
os.system(cmd)
# Save initial 3DGS attributes
gs_path = osp.join(root_path, 'gaussian_init')
os.makedirs(gs_path, exist_ok=True)
os.makedirs(osp.join(gs_path, 'smplx_init'), exist_ok=True)

cmd = f"mv {osp.join(root_path, 'images', '3dgs', 'shape_param.json')} {osp.join(gs_path, 'smplx_init')}"
os.system(cmd)
os.makedirs(osp.join(gs_path, '3dgs'), exist_ok=True)
cmd = f"mv {osp.join(root_path, 'images', '3dgs', '*.json')} {osp.join(gs_path, '3dgs')}"
print(cmd)
os.system(cmd)
cmd = f"rmdir {osp.join(root_path, 'images', '3dgs')}"
os.system(cmd)
os.makedirs(osp.join(gs_path, "smplx_optimized"), exist_ok=True)
os.makedirs(osp.join(gs_path, "smplx_optimized", "smplx_params"), exist_ok=True)
cmd = f'mv {MOTION_SEQ}/*.json {osp.join(gs_path, "smplx_optimized", "smplx_params")}'
print(cmd)
os.system(cmd)

# Extract views of canonical posed 3DGS

video_path = './exps/videos/video_human_benchmark/human-lrm-500M/neutral_rotation'
views_path = osp.join(gs_path, 'images')
os.makedirs(views_path, exist_ok=True)

cap = cv2.VideoCapture(glob(osp.join(video_path, '*.mp4'))[0])
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imwrite(os.path.join(views_path, f'{frame_idx}.png'), frame)
    frame_idx += 1
cap.release()

cmd = f'mv {glob(osp.join(video_path, "*.mp4"))[0]} {osp.join(gs_path, "images.mp4")}'
print(cmd)
os.system(cmd)

## Save keypoints_whole_body
kpt_set_coco = {
    'num': 133,
    'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
        *['Face_' + str(i) for i in range(52,69)], # face contour
        *['Face_' + str(i) for i in range(1,52)], # face
        'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
        'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
    }

kpt_set_smplx = {
    'num': 135, # 25 (body kpts) + 40 (hand kpts) + 70 (face keypoints)
    'name': ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body kpts
                'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand kpts
                'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand kpts
                'Head', 'Jaw', *['Face_' + str(i) for i in range(1,69)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
            ),
    'idx': (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body kpts
        37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand kpts
        52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand kpts
        15,22, # head, jaw
        76,77,78,79,80,81,82,83,84,85, # eyebrow
        86,87,88,89, # nose
        90,91,92,93,94, # below nose
        95,96,97,98,99,100,101,102,103,104,105,106, # eyes
        107, # right mouth
        108,109,110,111,112, # upper mouth
        113, # left mouth
        114,115,116,117,118, # lower mouth
        119, # right lip
        120,121,122, # upper lip
        123, # left lip
        124,125,126, # lower lip
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
        )
    }

from LHM.models.rendering.smplx import smplx
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
    model_path="./pretrained_models/human_model_files",
    model_type="smplx",
    gender="neutral",
    num_betas=10,
    num_expression_coeffs=100,
    use_pca=False,
    use_face_contour=True,
    flat_hand_mean=True,
    **layer_arg,
).cuda()

with open(osp.join(gs_path, 'smplx_init', 'shape_param.json'), 'r') as f: 
    shape_param = torch.FloatTensor(json.load(f)[0]).cuda()
smplx_param_list = glob(osp.join(gs_path, "smplx_optimized", "smplx_params", '*.json'))
smplx_param_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
os.makedirs(osp.join(gs_path, 'keypoints_whole_body'), exist_ok=True)
for smplx_param in smplx_param_list: 
    frame_idx = int(smplx_param.split('/')[-1].split('.')[0])
    with open(smplx_param, 'r') as f: 
        data = json.load(f)
        
    if "expr" not in data:
        expr = torch.zeros(100)
    else: 
        expr = data['expr']
        
    output = smplx_layer(global_orient=torch.FloatTensor(data['root_pose']).view(1,-1).cuda(),
                    body_pose=torch.FloatTensor(data['body_pose']).view(1,-1).cuda(),
                    jaw_pose=torch.FloatTensor(data['jaw_pose']).view(1,-1).cuda(),
                    leye_pose=torch.FloatTensor(data['leye_pose']).view(1,-1).cuda(),
                    reye_pose=torch.FloatTensor(data['reye_pose']).view(1,-1).cuda(),
                    left_hand_pose=torch.FloatTensor(data['lhand_pose']).view(1,-1).cuda(),
                    right_hand_pose=torch.FloatTensor(data['rhand_pose']).view(1,-1).cuda(),
                    expression=expr.view(1,-1).cuda(),
                    betas=shape_param.view(1,-1),
                    face_offset=None,
                    joint_offset=None)
    kpt_cam = output.joints[:,kpt_set_smplx['idx'],:]
    kpt_cam = kpt_cam + torch.FloatTensor(data['trans']).view(1,1,3).cuda()
    focal = torch.FloatTensor(data['focal']).cuda()
    princpt = torch.FloatTensor(data['princpt']).cuda()
    x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * focal[0] + princpt[0]
    y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * focal[1] + princpt[1]
    kpt_proj = torch.stack((x,y,torch.ones_like(x)),2).detach().cpu().numpy()[0]
    kpt_proj_save = change_kpt_name(kpt_proj, kpt_set_smplx['name'], kpt_set_coco['name'])
    with open(osp.join(gs_path, 'keypoints_whole_body', f'{frame_idx}.json'), 'w') as f: 
        json.dump(kpt_proj_save.tolist(), f)
    img = cv2.imread(osp.join(views_path, f'{frame_idx}.png'))
    for kpt in kpt_proj_save: 
        x_c, y_c = kpt[:2]
        cv2.circle(img, (int(x_c),int(y_c)), 1, (255,0,0), thickness=-1)
    cv2.imwrite(osp.join(gs_path, 'keypoints_whole_body', f'{frame_idx}.png'), img)

make_video(osp.join(gs_path, 'keypoints_whole_body.mp4'), osp.join(gs_path, 'keypoints_whole_body'))
    


# launch.py : python -m LHM.launch infer.human_lrm
#             REGISTRY_RUNNERS[args.runner] == HumanLRMInferrer in runners/infer/human_lrm.py!
#             runner.run() == HumanLRMInferrer.infer()

## inference pipeline!
# human_lrm.py : self.model: ModelHumanLRM
#                infer()
#                self.infer_single()      # or self.infer_mesh()
#                   motion_seq = prepare_motion_seqs(
#                   smplx_params = motion_seq['smplx_params']
#                   gs_model_list, query_points, transform_mat_neutral_pose = self.model.infer_single_view(..., smplx_params, ...)
#                   batch_smplx_params[key] = motion_seq...    # motion sequence 를 배치로 가져옴

## Model module!
# LHM/models/modeling_human_lrm.py
#   class ModelHumanLRMSapdinoBodyHeadSD3_5
#       self.renderer: GS3DRenderer
#       def infer_single_view(image, smplx_params, ...))
#           gs_model_list, query_points, smplx_params = self.renderer.forward_gs(
#           return out == dict{render_rgb, render_mask, render_depth}


## 3DGS module!
# LHM/models/rendering/gs_renderer.py
#     class GS3DRenderer
#         self.forward_gs_attr
#             gs_attr = self.forward_gs_attr()
#                 return gs_attr: GaussianAppOutput
#             gs_attr_list.append(gs_attr)
#             return gs_attr_list, ...

## 3DGS attribute format!
# LHM/outputs/output.py
#     class GaussianAppOutput(BaseOutput):
'''
offset_xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor
    use_rgb: bool
'''


# class GS3DRenderer
#   positions, ... = self.smplx_model.get_query_points(smplx_data, device=device)
# query_points ==  smplx mesh_neutral_pose (upsampled)  (canonical(大) pose)