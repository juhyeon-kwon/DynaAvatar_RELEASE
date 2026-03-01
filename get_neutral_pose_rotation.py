import torch
import argparse
import os
import os.path as osp
import json

from LHM.models.rendering.smpl_x import SMPLX
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, dest='save_path')
    parser.add_argument('--view_num', type=str, dest='view_num')
    args = parser.parse_args()

    return args

def main():
    # argument parse and create log
    args = parse_args()
    save_path = args.save_path
    view_num = int(args.view_num)

    os.makedirs(save_path, exist_ok=True)
    
    human_model_path = "./pretrained_models/human_model_files"
    smpl_x = SMPLX(
        human_model_path=human_model_path,
        cano_pose_type=1)
    
    render_shape = [1024, 1024]
    zero_pose = torch.zeros((3)).float()
    smplx_param = {'root_pose': torch.FloatTensor([math.pi,0,0]), \
                'body_pose': smpl_x.neutral_body_pose, \
                'jaw_pose': zero_pose, \
                'leye_pose': zero_pose, \
                'reye_pose': zero_pose, \
                'lhand_pose': torch.zeros((len(smpl_x.joint_part['lhand']),3)).float(), \
                'rhand_pose': torch.zeros((len(smpl_x.joint_part['rhand']),3)).float(), \
                'trans': torch.FloatTensor((0,0,4)).float()}
    
    lhm_smplx_params = {}
    for key, val in smplx_param.items():
        lhm_smplx_params[key] = val.tolist()
    
    lhm_smplx_params['betas'] = torch.zeros([10]).float().tolist()
    lhm_smplx_params['focal'] = torch.FloatTensor((1500,1500)).tolist()
    lhm_smplx_params['princpt'] = torch.FloatTensor((render_shape[1]/2, render_shape[0]/2)).tolist()
    lhm_smplx_params['img_size_wh'] = render_shape
    lhm_smplx_params['pad_ratio'] = 0.0
    
    root_rot1 = torch.FloatTensor([math.pi,0,0])
    root_rot1 = axis_angle_to_matrix(root_rot1)
    for i, elev in enumerate([-math.pi/6, 0, math.pi/6]): 
        root_rot2 = torch.FloatTensor([elev,0,0])
        root_rot2 = axis_angle_to_matrix(root_rot2)
        for j in range(view_num): 
            angle = math.pi * 2 * j / view_num
            root_rot3 = torch.FloatTensor([0,angle,0])  # azim
            root_rot3 = axis_angle_to_matrix(root_rot3)
            root_pose = torch.matmul(root_rot1, root_rot2)
            root_pose = torch.matmul(root_pose, root_rot3)
            root_pose = matrix_to_axis_angle(root_pose)
            lhm_smplx_params['root_pose'] = root_pose.tolist()
            
            with open(osp.join(save_path, f'{i*view_num+j:05d}.json'), 'w') as f:
                json.dump(lhm_smplx_params, f)
    
    '''for i in range(view_num):
        azim = math.pi + math.pi*2*i/view_num # azim angle of the camera
        elev = -math.pi/6 
        dist = torch.sqrt(torch.sum((cam_pos - at_point)**2)) # distance between camera and mesh
        R, t = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=False, at=at_point[None,:], up=((0,1,0),)) 
        R = torch.inverse(R)
        cam_param_rot = {'R': R[0].cuda(), 't': t[0].cuda(), 'focal': cam_param['focal'], 'princpt': cam_param['princpt']}'''

    
    print(f"Saved neutral pose smplx params to {save_path}")



if __name__ == "__main__":
    main()