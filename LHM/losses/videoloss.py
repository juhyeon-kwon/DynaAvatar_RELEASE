import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan

__all__ = ['WanVAELoss']

# implemented by vcai
class WanVAELoss(nn.Module):

    def __init__(self, device, model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"):
        super().__init__()
        self.device = device
        self.model_id = model_id
        
        self.vae = None
        
        self.loss_fn = nn.MSELoss()

    def _load_vae_if_needed(self):
        if self.vae is None:
            #print("Loading Wan VAE for perceptual loss...")

            self.vae = AutoencoderKLWan.from_pretrained(
                self.model_id, subfolder="vae"
            ).to(self.device)
            
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

    def forward(self, rendered_seq, gt_seq):
        """        
        Args: 
            rendered_seq (torch.Tensor): Shape: [B, C, F, H, W]
            gt_seq (torch.Tensor): Shape: [B, C, F, H, W]
        
        Returns:
            torch.Tensor: 계산된 스칼라 손실 값.
        """
        self._load_vae_if_needed()

        # normalize [-1, 1]
        rendered_seq = rendered_seq * 2.0 - 1.0
        gt_seq = gt_seq * 2.0 - 1.0

        B, F, V, C, H, W = rendered_seq.shape

        # (B, F, V, C, H, W) -> (B*V, C, F, H, W)
        rendered_seq = rendered_seq.permute(0, 2, 1, 3, 4, 5).reshape(B * V, C, F, H, W)
        gt_seq = gt_seq.permute(0, 2, 1, 3, 4, 5).reshape(B * V, C, F, H, W)

        with torch.no_grad():
            posterior_x = self.vae.encode(rendered_seq).latent_dist
            posterior_y = self.vae.encode(gt_seq).latent_dist

        mean_x = posterior_x.mean
        mean_y = posterior_y.mean

        loss = self.loss_fn(mean_x, mean_y)
        return loss