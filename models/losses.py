# models/losses.py
import torch
import torch.nn as nn

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