import torch
import torch.nn as nn
import torchvision.transforms.functional as F

def crop_rendered_face_rgb(img, bbox_t, target_size, align_corners=True): # (B'=B*T, Nv, 3, H, W), (B'=B*T, 4)
    target_h, target_w = target_size
    B, Nv, C, H, W = img.shape
    img_ = img.clone().reshape(-1,C,H,W)
    #bbox_t = bbox_t.repeat(Nv, 1)  # (B*T,4)  # TODO Nv
    
    x0s, y0s, x1s, y1s = bbox_t.unbind(1)  # (B*T)
    
    # align_corners=True based
    x0s = x0s.clamp(0, W-1)
    y0s = y0s.clamp(0, H-1)
    x1s = x1s.clamp(1, W)
    y1s = y1s.clamp(1, H)
    
    x0s = 2.0 * x0s / (W - 1) - 1.0
    x1s = 2.0 * (x1s - 1) / (W - 1) - 1.0
    y0s = 2.0 * y0s / (H - 1) - 1.0
    y1s = 2.0 * (y1s - 1) / (H - 1) - 1.0
    
    sxs = (x1s - x0s) / 2.0
    txs = (x1s + x0s) / 2.0
    sys = (y1s - y0s) / 2.0
    tys = (y1s + y0s) / 2.0
    
    theta = torch.zeros(B, 2, 3, device=img_.device, dtype=img_.dtype)
    theta[:, 0, 0] = sxs
    theta[:, 0, 2] = txs
    theta[:, 1, 1] = sys
    theta[:, 1, 2] = tys
    
    # Make batch grid (128, 128) & sampling
    grid = nn.functional.affine_grid(theta, size=(B, C, target_h, target_w), align_corners=align_corners)
    crops = nn.functional.grid_sample(img_, grid, mode="bilinear", align_corners=align_corners)  # (B',3,target_h,target_w)
    crops = crops.reshape(B, Nv, C, target_h, target_w)

    return crops # B, Nv, 3, 128, 128


class ArcFaceLoss(nn.Module): 
    def __init__(self, face_embedder, device='cuda'): 
        super(ArcFaceLoss, self).__init__()
        self.face_embedder = face_embedder
    
    def forward(self, out_img, target_face_img, target_face_bbox, mask, DATA, target_size=(128,128)):  # (B, Nv, 3, H, W), (B, Nv, 3, 128, 128), (B, 4)
        target_h, target_w = target_size
        B = out_img.shape[0]
        
        # face img from out_img
        out_img_face = crop_rendered_face_rgb(out_img, target_face_bbox, target_size=target_size) # (B, Nv, 3, 128, 128)
        
        # rgb to grayscale       
        out_img_face_gray = F.rgb_to_grayscale(out_img_face)  # (B, Nv, 1, 128, 128)
        target_face_gray = F.rgb_to_grayscale(target_face_img)  # (B, Nv, 1, 128, 128)
        
        # debug
        '''if DATA['is_target_face_detected'][0]: 
            single_rgb_image = out_img_face[0, 0]
            import numpy as np
            import cv2
            rgb_image_np = (single_rgb_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            rgb_image_bgr = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{DATA['uid'][0]}_debug_rgb_face.png", rgb_image_bgr)
            print(f"RGB Saved: {DATA['uid'][0]}_debug_rgb_face.png")
            
            single_gray_image = out_img_face_gray[0, 0]
            gray_image_np = (single_gray_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            #rgb_image_bgr = cv2.cvtColor(gray_image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{DATA['uid'][0]}_debug_gray_face.png", gray_image_np)
            print(f"RGB Saved: {DATA['uid'][0]}_debug_gray_face.png")
            
            print("target face img shape:", target_face_img.shape)
            single_gray_image_target = target_face_gray[0, 0]
            gray_image_np_target = (single_gray_image_target.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            #rgb_image_bgr = cv2.cvtColor(gray_image_np_target, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{DATA['uid'][0]}_debug_gray_face_gt.png", gray_image_np_target)
            print(f"RGB Saved: {DATA['uid'][0]}_debug_gray_face_gt.png")'''
        
        # [-1, 1] normalization
        out_face = out_img_face_gray * 2 - 1.
        target_face = target_face_gray * 2 - 1.
        out_face_embed = self.face_embedder(out_face.reshape(-1, 1, target_h, target_w))  # (B*Nv, 512)
        target_face_embed = self.face_embedder(target_face.reshape(-1, 1, target_h, target_w))
        
        # normalize
        out_face_embed = nn.functional.normalize(out_face_embed, p=2, dim=-1)
        target_face_embed = nn.functional.normalize(target_face_embed, p=2, dim=-1)
        
        # L1 loss, L2 loss?
        weight = mask.float()  # (B*Nv,)

        embed_loss = (out_face_embed - target_face_embed).abs().sum(dim=-1) * weight  # (B*Nv,)
        batch_loss = embed_loss.reshape(B, -1).mean(dim=1)
        all_loss = batch_loss.mean()
        
        return all_loss