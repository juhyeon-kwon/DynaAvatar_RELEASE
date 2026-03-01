# Copyright (c) 2023-2024, Zexin He
#
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


import torch
import torch.nn as nn


class CameraEmbedder(nn.Module):
    """
    Embed camera features to a high-dimensional vector.
    
    Reference:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L27
    """
    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.compile
    def forward(self, x):
        return self.mlp(x)

import math
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=17):
        """
        Args:
            embedding_dim (int): 임베딩 차원의 크기. body_features의 마지막 차원 크기와 같아야 합니다.
            max_len (int): 예상되는 시퀀스의 최대 길이.
        """
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        
        # 10000^(2i/d_model) 
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_len, embedding_dim)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape: (B, T, D)
        
        Returns:
            torch.Tensor: Shape: (B, T, D)
        """
        # x의 시퀀스 길이(T)만큼만 positional encoding을 잘라내서 더함
        # self.pe -> (1, max_len, D) 이므로, x -> (B, T, D)에 broadcasting되어 더해짐
        x = x + self.pe[:, :x.size(1)]
        return x