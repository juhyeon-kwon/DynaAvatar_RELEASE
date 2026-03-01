import torch
import torch.nn as nn
from lightglue import SuperPoint, LightGlue
import torch.nn.functional as F

# --- 디버깅에 필요한 라이브러리 (matplotlib, numpy, os) ---
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection # plot_matches_replacement를 위해 import
import random
# vvvvvvvvvvvv lightglue.viz.plot_matches 대체 함수 vvvvvvvvvvvvvv
def plot_matches_replacement(ax, img0_np, img1_np, kpts0_np, kpts1_np, matches_np, a=0.1):
    """
    matplotlib을 사용해 plot_matches를 흉내 내는 함수
    """
    # 1. 두 이미지를 좌우로 붙이기 (회색조 이미지도 RGB로 통일)
    if img0_np.ndim == 2:
        img0_np = np.stack([img0_np] * 3, axis=-1)
    if img1_np.ndim == 2:
        img1_np = np.stack([img1_np] * 3, axis=-1)
        
    H0, W0, _ = img0_np.shape
    H1, W1, _ = img1_np.shape
    
    # 높이가 다를 경우, 더 큰 쪽에 맞춰 회색 배경 생성
    H_max = max(H0, H1)
    W_max = W0 + W1
    
    img_combined = np.full((H_max, W_max, 3), 128, dtype=np.uint8)
    img_combined[:H0, :W0, :] = img0_np
    img_combined[:H1, W0:W0+W1, :] = img1_np
    
    ax.imshow(img_combined)
    
    # 2. 매칭된 키포인트 좌표 가져오기
    pts0 = kpts0_np[matches_np[:, 0]]
    pts1 = kpts1_np[matches_np[:, 1]]
    
    # 3. 선 그리기
    # pts1의 x좌표는 W0만큼 이동시켜야 함
    pts1_shifted = pts1.copy()
    pts1_shifted[:, 0] += W0
    
    # LineCollection을 사용해 효율적으로 선 그리기
    lines = np.stack([pts0, pts1_shifted], axis=1) # (N, 2, 2) 형태
    
    # 랜덤 색상 생성 (alpha 적용)
    colors = np.random.rand(len(lines), 3)
    colors = np.hstack([colors, np.full((len(lines), 1), a)])
    
    lc = LineCollection(lines, colors=colors, linewidths=0.5)
    ax.add_collection(lc)
    
    ax.axis('off') # 축 끄기
# ^^^^^^^^^^^^^^ plot_matches 대체 함수 ^^^^^^^^^^^^^^


class FeatureMatchingLoss(nn.Module):
    def __init__(self, max_num_keypoints=1024, device='cuda'):
        super(FeatureMatchingLoss, self).__init__()
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        # --- 디버깅용: 출력 디렉토리 설정 ---
        self.debug_output_dir = "debug_viz_output"
        if not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir)
        print(f"[Debug] 시각화 파일은 {self.debug_output_dir} 에 저장됩니다.")
        # ------------------------------------

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
                print(f"[Debug] Frame {i}: 매칭 없음, 스킵.")
                continue

            # (원본 코드) 키포인트 좌표 및 그리드 샘플링
            matched_kpts_out_pixels = kpts_out[matches[:, 0]] # 범위 [0, S-1]
            
            # align_corners=False 에 맞는 그리드 정규화
            # 픽셀 좌표 [0, W-1] (픽셀 센터) -> NDC [-1, 1]
            # 공식: ( (pix_coord + 0.5) / S ) * 2.0 - 1.0
            grid_x = ((matched_kpts_out_pixels[:, 0] + 0.5) / W) * 2.0 - 1.0
            grid_y = ((matched_kpts_out_pixels[:, 1] + 0.5) / H) * 2.0 - 1.0
            sampling_grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(0).unsqueeze(0)


            # vvvvvvvvvvvv [핵심 수정] Un-premultiply vvvvvvvvvvvvvv
            
            # 1. Premultiplied 좌표 (U*A, V*A) 샘플링 (mode='bilinear')
            sampled_premult_coords = F.grid_sample(
                proj_frame[:, :2, :, :],  # R, G 채널 (U*A, V*A)
                sampling_grid, 
                mode='bilinear',
                align_corners=False,
                padding_mode='zeros'
            ).view(2, -1).T # [N_matches, 2]

            # 2. Alpha (A) 샘플링
            sampled_alpha = F.grid_sample(
                proj_alpha, # 알파 채널 (A)
                sampling_grid,
                mode='bilinear',
                align_corners=False,
                padding_mode='zeros'
            ).view(1, -1).T # [N_matches, 1]

            # 3. Un-premultiply (좌표 복원)
            epsilon = 1e-6 # 0으로 나누기 방지
            
            # (U*A) / (A + epsilon) = U
            # (V*A) / (A + epsilon) = V
            sampled_proj_coords = sampled_premult_coords / (sampled_alpha + epsilon)
            
            # ^^^^^^^^^^^^^^ [핵심 수정] 끝 ^^^^^^^^^^^^^^


            # 타겟 좌표
            matched_kpts_target_pixels = kpts_target[matches[:, 1]] # 범위 [0, S-1]
            
            # (align_corners=False 에 맞게 [0, 1]로 정규화)
            target_coords_normalized = torch.stack([
                (matched_kpts_target_pixels[:, 0] + 0.5) / W,
                (matched_kpts_target_pixels[:, 1] + 0.5) / H
            ], dim=1)


            # vvvvvvvvvvvv [신규] grid_sample 디버깅 (비교) vvvvvvvvvvvvvv
            print(f"  [Target] target_coords_normalized:")
            print(f"    (True U) min/max/mean: {target_coords_normalized[:,0].min():.2f}, {target_coords_normalized[:,0].max():.2f}, {target_coords_normalized[:,0].mean():.2f}")
            print(f"    (True V) min/max/mean: {target_coords_normalized[:,1].min():.2f}, {target_coords_normalized[:,1].max():.2f}, {target_coords_normalized[:,1].mean():.2f}")
            print(f"-------------------------------------------")
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


            # vvvvvvvvvvvvvv 디버깅 시각화 코드 (최종 수정본 9 - Overlay) vvvvvvvvvvvvvv
            
            print(f"--- DEBUGGING FRAME {i} (매칭 {matches.shape[0]}개 발견) ---")
            
            # 1. 텐서를 NumPy로 변환 (CPU로 이동)
            with torch.no_grad():
                # [수정] 두 패널의 배경 이미지를 모두 로드
                out_img_np = out_frame[0].permute(1, 2, 0).cpu().numpy()
                target_img_np = target_frame[0].permute(1, 2, 0).cpu().numpy()
                
                # C=1 이면 (H, W)로, C=3 이면 (H, W, 3)으로
                if out_img_np.ndim == 2 or out_img_np.shape[2] == 1:
                    out_img_np = out_img_np.squeeze()
                    out_img_np = np.stack([out_img_np] * 3, axis=-1) # RGB로 통일
                if target_img_np.ndim == 2 or target_img_np.shape[2] == 1:
                    target_img_np = target_img_np.squeeze()
                    target_img_np = np.stack([target_img_np] * 3, axis=-1) # RGB로 통일

                if out_img_np.max() <= 1.0:
                    out_img_np = (out_img_np * 255).astype(np.uint8)
                if target_img_np.max() <= 1.0:
                    target_img_np = (target_img_np * 255).astype(np.uint8)
                
                # (알파 맵은 실루엣에만 사용했으므로 제거)
                # proj_alpha_np = proj_alpha[0, 0, ...].cpu().numpy() 

                # 3. 플롯 생성 (1행 2열)
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                
                # H, W는 이전에 '_, _, H, W = proj_frame.shape'로 정의됨
                
                # 좌표 변환 (Overlay에 필요)
                query_pts_px = matched_kpts_out_pixels.cpu().numpy()
                true_pts_px = (target_coords_normalized.detach().cpu().numpy() * [W-1, H-1])
                
                # 알파 값으로 필터링 마스크 생성
                alpha_threshold = 0.9
                sampled_alpha_np = sampled_alpha.detach().cpu().numpy().squeeze()
                valid_mask_np = (sampled_alpha_np > alpha_threshold)

                print(f"    [Filtering] 총 {len(valid_mask_np)}개 샘플 중 {valid_mask_np.sum()}개 통과 (Alpha > {alpha_threshold})")

                # 유효한(Valid) 좌표만 미리 필터링
                valid_queries = query_pts_px[valid_mask_np]
                valid_targets = true_pts_px[valid_mask_np]
                
                # 3-1. [좌측 패널] Rendered(Out) 이미지 + Flow Overlay
                axes[0].imshow(out_img_np)
                axes[0].axis('off')
                
                # 3-2. [우측 패널] GT(Target) 이미지 + Flow Overlay
                axes[1].imshow(target_img_np)
                axes[1].axis('off')

                # 두 패널 모두에 동일한 점과 화살표를 그립니다.
                for ax in axes:
                    # 유효한 점(valid points)만 흰색 점으로 찍기
                    ax.scatter(valid_queries[:, 0], valid_queries[:, 1], c='white', edgecolors='black', linewidths=0.5, s=12, zorder=3)
                    


                    # 유효한 화살표(valid arrows)만 순회하며 그리기
                    for k in range(len(valid_queries)):
                        q = valid_queries[k]
                        t = valid_targets[k]
                        
                        dx = (t[0]-q[0]) * 1.1
                        dy = (t[1]-q[1]) * 1.1

                        # (녹색 GT 화살표) - 가시성 확보
                        ax.arrow(q[0], q[1], dx, dy, 
                                  color='red', 
                                  width=1.0,           # 몸통 두께
                                  head_width=5.0,      # 머리 너비
                                  length_includes_head=True, 
                                  alpha=1.0, 
                                  zorder=2)
                
                # 4. 파일로 저장
                pid = os.getpid()
                save_path = os.path.join(self.debug_output_dir, f"debug_frame_{i}_pid{pid+random.randint(1, 500)}.png")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_path)
                print(f"시각화 파일 저장 완료: {save_path}")
                plt.close(fig)

                # 5. 프로그램 종료 (주석 처리됨)
                #print("--- 디버깅 시각화 저장 완료. 요청대로 프로그램을 종료합니다. ---")
                #exit()
                
            # ^^^^^^^^^^^^^^ 디버깅 시각화 코드 (최종 수정본 9 - Overlay) ^^^^^^^^^^^^^^


            # vvvvvvvvvvvv [핵심 수정] 로스 계산 (필터링 적용) vvvvvvvvvvvvvv
            
            # 1. 필터링 마스크 생성
            alpha_threshold = 0.9
            valid_mask = (sampled_alpha > alpha_threshold).squeeze() # [N_matches]
            
            if valid_mask.sum() == 0:
                print(f"[Debug] Frame {i}: 유효한 샘플링 없음 (알파가 너무 낮음), 스킵.")
                continue

            # 2. 유효한 샘플, 타겟, 스코어만 추출
            loss_coords = sampled_proj_coords[valid_mask]
            target_coords = target_coords_normalized[valid_mask]
            valid_scores = scores[valid_mask].unsqueeze(1) # [N_valid, 1]

            # 3. 필터링된 로스 계산
            confidence_weighted_frame_loss = ((loss_coords - target_coords) ** 2) * valid_scores
            
            # print(proj_out_mask.shape, confidence_weighted_frame_loss.shape) # <--- 삭제된 print
            
            avg_loss = torch.sum(confidence_weighted_frame_loss) / (torch.sum(valid_scores) + 1e-6)
            
            total_loss += avg_loss
            num_frames_with_matches += 1

        if num_frames_with_matches == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / num_frames_with_matches