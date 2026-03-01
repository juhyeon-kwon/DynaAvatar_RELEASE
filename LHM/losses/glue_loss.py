import torch
import torch.nn as nn
from lightglue import SuperPoint, LightGlue
import torch.nn.functional as F
import os 

class FeatureMatchingLoss(nn.Module):
    def __init__(self, max_num_keypoints=1024, device='cuda'):
        super(FeatureMatchingLoss, self).__init__()
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)


    def forward(self, out_video, target_video, proj_out_video, proj_out_mask):
        B_T, V, C, H, W = out_video.shape
        out_video = out_video.reshape(B_T * V, C, H, W)
        target_video = target_video.reshape(B_T * V, C, H, W)
        proj_out_video = proj_out_video.reshape(B_T * V, C, H, W) # C=3 가정
        proj_out_mask = proj_out_mask.reshape(B_T * V, 1, H, W) # C=1 가정

        batch_size = out_video.shape[0]
        total_loss = 0.0
        num_frames_with_matches = 0

        for i in range(batch_size):
            out_frame = out_video[i:i+1]
            target_frame = target_video[i:i+1]
            proj_frame = proj_out_video[i:i+1] # [1, 3, H, W]
            proj_alpha = proj_out_mask[i:i+1] # [1, 1, H, W]
            _, _, H, W = proj_frame.shape

            with torch.no_grad():
                feats_out = self.extractor.extract(out_frame.to(self.device))
                feats_target = self.extractor.extract(target_frame.to(self.device))
                matches_data = self.matcher({'image0': feats_out, 'image1': feats_target})

                kpts_out = feats_out['keypoints'][0]
                kpts_target = feats_target['keypoints'][0]
                matches = matches_data['matches'][0]
                scores = matches_data['scores'][0]

            if matches.shape[0] == 0:
                continue

            # 1. 샘플링 그리드 준비 (NDC 좌표)
            matched_kpts_out_pixels = kpts_out[matches[:, 0]] # 범위 [0, S-1]
            grid_x = ((matched_kpts_out_pixels[:, 0] + 0.5) / W) * 2.0 - 1.0
            grid_y = ((matched_kpts_out_pixels[:, 1] + 0.5) / H) * 2.0 - 1.0
            sampling_grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(0).unsqueeze(0)


            # 2. [핵심] Un-premultiply (사전 곱셈 해제)
            
            # 2a. Premultiplied 좌표 (U*A, V*A) 샘플링
            sampled_premult_coords = F.grid_sample(
                proj_frame[:, :2, :, :],  # R, G 채널 (U*A, V*A)
                sampling_grid, 
                mode='bilinear',
                align_corners=False,
                padding_mode='zeros'
            ).view(2, -1).T # [N_matches, 2]

            # 2b. Alpha (A) 샘플링
            sampled_alpha = F.grid_sample(
                proj_alpha, # 알파 채널 (A)
                sampling_grid,
                mode='bilinear',
                align_corners=False,
                padding_mode='zeros'
            ).view(1, -1).T # [N_matches, 1]

            # 2c. 좌표 복원: (U*A) / (A + epsilon) = U
            epsilon = 1e-6 
            sampled_proj_coords = sampled_premult_coords / (sampled_alpha + epsilon)
            
            # 3. 타겟 좌표 준비
            matched_kpts_target_pixels = kpts_target[matches[:, 1]]
            target_coords_normalized = torch.stack([
                (matched_kpts_target_pixels[:, 0] + 0.5) / W,
                (matched_kpts_target_pixels[:, 1] + 0.5) / H
            ], dim=1)


            alpha_threshold = 0.9
            valid_mask = (sampled_alpha > alpha_threshold).squeeze() # [N_matches]
            
            if valid_mask.sum() == 0:
                continue

            # 4b. 유효한 샘플, 타겟, 스코어만 추출
            loss_coords = sampled_proj_coords[valid_mask]
            target_coords = target_coords_normalized[valid_mask]
            valid_scores = scores[valid_mask].unsqueeze(1) # [N_valid, 1]

            # 4c. 필터링된 로스 계산 (L2 Loss)
            #confidence_weighted_frame_loss = ((loss_coords - target_coords) ** 2) * valid_scores
            
            #ㅣ1
            confidence_weighted_frame_loss = (torch.abs(loss_coords - target_coords)) * valid_scores


            # 4d. 가중 평균
            avg_loss = torch.sum(confidence_weighted_frame_loss) / (torch.sum(valid_scores) + 1e-6)
            
            total_loss += avg_loss
            num_frames_with_matches += 1

        if num_frames_with_matches == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / num_frames_with_matches