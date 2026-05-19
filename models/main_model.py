import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layers.GraphConvolution import GraphConvolution
from models.layers.UE import UE

class TCIF_LPUE(nn.Module):
    def __init__(self, adj_matrix):
        super(TCIF_LPUE, self).__init__()
        
        self.register_buffer('adj', torch.tensor(adj_matrix, dtype=torch.float32))

        self.conv1_img = nn.Conv2d(10, 64, kernel_size=7, padding=3, stride=2)
        self.maxpool_img = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_img = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_img = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_img = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1_img = nn.Linear(256, 169)
        self.linear2_img = nn.Linear(169, 169)
        
        self.conv1_phy = nn.Conv2d(5, 64, kernel_size=7, padding=3, stride=2)
        self.conv2_phy = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_phy = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_phy = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.att_block_1 = self._build_spatial_gate(64)
        self.att_block_2 = self._build_spatial_gate(128)
        self.att_block_3 = self._build_spatial_gate(256)

        self.cross_unit = nn.Parameter(data=torch.ones(9))
        self.fuse_unit = nn.Parameter(data=torch.ones(6))
        
        self.linear1_phy = nn.Linear(256, 169)
        self.linear2_phy = nn.Linear(169, 169)

        self.gcn1 = GraphConvolution(in_features=5, out_features=8)
        self.gcn2 = GraphConvolution(in_features=8, out_features=16)
        
        self.gcn_projection = nn.Linear(9 * 16, 20)
        self.ue_his_projection = nn.Linear(9 * 16, 5 * 32)

        self.UE = UE()
        
        self.linear_mean = nn.Linear(169 + 169 + 20, 1)
        self.linear_var = nn.Linear(169 + 169 + 20, 1)

    def _build_spatial_gate(self, dim):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

    def forward(self, x_img, x_ERA, x_onehot):
        x_img = self.maxpool_img(F.relu(self.conv1_img(x_img)))
        x_img = self.maxpool_img(F.relu(self.conv2_img(x_img)))
        x_img = self.maxpool_img(F.relu(self.conv3_img(x_img)))
        
        x_img_pool = self.maxpool_img(x_img) 
        input_image = x_img_pool.permute(0, 2, 3, 1).contiguous()
        
        x_img = self.global_pool(F.relu(self.conv4_img(x_img))).view(-1, 256)
        f_img = F.relu(self.linear2_img(F.relu(self.linear1_img(x_img))))
        
        v1, v2, v3 = x_ERA[:, 0], x_ERA[:, 1], x_ERA[:, 2]

        def _extract_feature(tensors, conv_op, gate_op):
            return [gate_op(self.maxpool_img(F.relu(conv_op(t)))) * self.maxpool_img(F.relu(conv_op(t))) for t in tensors]

        def _adaptive_blend(feature_list, weights, start_idx, count):
            w_slice = weights[start_idx : start_idx + count]
            w_sum = w_slice.sum()
            blended = 0
            for i, feat in enumerate(feature_list):
                blended = blended + (w_slice[i] / w_sum) * feat
            return blended

        f1, f2, f3 = _extract_feature([v1, v2, v3], self.conv1_phy, self.att_block_1)
        mixed_1 = _adaptive_blend([f1, f2, f3], self.cross_unit, 0, 3)
        
        mixed_1_out = _extract_feature([mixed_1], self.conv2_phy, self.att_block_2)[0]
        f1, f2, f3 = _extract_feature([f1, f2, f3], self.conv2_phy, self.att_block_2)
        mixed_2 = _adaptive_blend([f1, f2, f3], self.cross_unit, 3, 3)
        
        fused_1 = _adaptive_blend([mixed_1_out, mixed_2], self.fuse_unit, 0, 2)
        
        fused_1_out = _extract_feature([fused_1], self.conv3_phy, self.att_block_3)[0]
        f1, f2, f3 = _extract_feature([f1, f2, f3], self.conv3_phy, self.att_block_3)
        mixed_3 = _adaptive_blend([f1, f2, f3], self.cross_unit, 6, 3)
        
        fused_2 = _adaptive_blend([fused_1_out, mixed_3], self.fuse_unit, 2, 2)
        
        out_phy = F.relu(self.conv4_phy(fused_2)) 
        f_phy = self.global_pool(out_phy).view(-1, 256)
        f_phy = F.relu(self.linear2_phy(F.relu(self.linear1_phy(f_phy))))

        gcn1_out = F.relu(self.gcn1(x_onehot, self.adj))
        gcn2_out = F.relu(self.gcn2(gcn1_out, self.adj))
        gcn_flat = gcn2_out.view(-1, 9 * 16)
        
        f_hist = F.relu(self.gcn_projection(gcn_flat))
        input_his = self.ue_his_projection(gcn_flat).view(-1, 5, 32)
        
        f_fused = torch.cat((f_img, f_phy, f_hist), dim=1)
        y_mean = self.linear_mean(f_fused)
        y_log_var = self.linear_var(f_fused)

        loss_ue_img, loss_ue_his, _ = self.UE(input_image, input_his)
        
        return y_mean, y_log_var, loss_ue_img, loss_ue_his

class WeightedLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedLoss, self).__init__()
        self.weights = weights

    def forward(self, losses):
        device = losses[0].device
        weights_tensor = torch.tensor(self.weights, dtype=torch.float32, device=device)
        return torch.sum(torch.stack(losses) * weights_tensor)

class NegativeLogLikelihoodLoss(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihoodLoss, self).__init__()

    def forward(self, y_pred_mean, y_log_var, y_true):
        y_log_var = torch.clamp(y_log_var, min=-10.0, max=10.0)
        loss = 0.5 * torch.exp(-y_log_var) * ((y_true - y_pred_mean) ** 2) + 0.5 * y_log_var
        return torch.sum(loss)