import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

def get_neighbor(vertex_num, face, neighbor_max_num=10):
    adj = {i: set() for i in range(vertex_num)}
    for i in range(len(face)):
        for idx in face[i]:
            adj[idx] |= set(face[i]) - set([idx])

    neighbor_idxs = np.tile(np.arange(vertex_num)[:,None], (1, neighbor_max_num))
    neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)
    for idx in range(vertex_num):
        neighbor_num = min(len(adj[idx]), neighbor_max_num)
        neighbor_idxs[idx,:neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]
        neighbor_weights[idx,:neighbor_num] = -1.0 / neighbor_num
    
    neighbor_idxs, neighbor_weights = torch.from_numpy(neighbor_idxs), torch.from_numpy(neighbor_weights)
    return neighbor_idxs, neighbor_weights

from pytorch3d.ops import knn_points

def get_neighbor_knn(vertices, k=10):
    verts_batch = vertices.unsqueeze(0)
    
    # 각 점(verts_batch)에 대해 가장 가까운 k+1개의 점을 찾습니다.
    # (자기 자신 포함, 따라서 k+1)
    _, neighbor_idxs, _ = knn_points(verts_batch, verts_batch, K=k + 1, return_nn=False)
    
    neighbor_idxs = neighbor_idxs.squeeze(0)[:, 1:] # (N, k)
    
    # 가중치는 기존 방식과 동일하게 -1/k 로 설정합니다.
    neighbor_weights = torch.full_like(neighbor_idxs, -1.0 / k, dtype=torch.float32)
    
    return neighbor_idxs, neighbor_weights


class LaplacianRegLoss(nn.Module):
    def __init__(self, canonical_vertices, k=10):
        super(LaplacianRegLoss, self).__init__()
        neighbor_idxs, neighbor_weights = get_neighbor_knn(canonical_vertices, k=k)
        self.neighbor_idxs, self.neighbor_weights = neighbor_idxs.cuda(), neighbor_weights.cuda()


    def compute_laplacian(self, x, neighbor_idxs, neighbor_weights):
        lap = x + (x[:, neighbor_idxs] * neighbor_weights[None, :, :, None]).sum(2)
        return lap

    def forward(self, out, target):
        if target is None:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            loss = lap_out ** 2
            return loss
        else:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            lap_target = self.compute_laplacian(target, self.neighbor_idxs, self.neighbor_weights)
            loss = (lap_out - lap_target) ** 2
            return loss
            