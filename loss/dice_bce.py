import torch.nn as nn
import torch
from loss.dice import DiceLoss

class DBCE(nn.Module):
    def __init__(self, W_s=1, W_b=5, W_f=2):
        super(DBCE, self).__init__()
        self.W_s = W_s  # 1 mirror maps
        self.W_b = W_b  # 5 edge maps
        self.W_f = W_f  # 2 final maps
        self.BCEwithLogitsLoss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target, target_edge):
        num_map = len(pred)
        spec_map_loss = 0
        edge_loss = 0
        final_loss = 0
        
        for i in range(num_map):
            if i < num_map - 2:
                spec_map_loss += self.W_s * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
            elif i == num_map - 2:
                edge_loss = self.W_b * self.BCEwithLogitsLoss(pred[i], target_edge)
            elif i == num_map - 1:
                final_loss = self.W_f * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
                
        return spec_map_loss, edge_loss, final_loss


class LossComponent(nn.Module):
    def __init__(self, W_map=1, W_final=2):
        super(LossComponent, self).__init__()
        self.W_map = W_map
        self.W_final = W_final
        self.BCEwithLogitsLoss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target, target_edge):
        sum_loss = 0
        for i in range(len(pred)):
            if i < len(pred) - 1:
                sum_loss += self.W_map * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
            else:
                sum_loss += self.W_final * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
        return sum_loss


class LossRefine(nn.Module):
    def __init__(self, W_map=5, W_final=2):
        super(LossRefine, self).__init__()
        self.W_map = W_map
        self.W_final = W_final
        self.BCEwithLogitsLoss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target, target_edge):
        edge_loss = self.W_map * self.BCEwithLogitsLoss(pred[-2], target_edge)
        final_loss = self.W_final * (self.BCEwithLogitsLoss(pred[-1], target) + self.dice_loss(pred[-1], target))
        return edge_loss + final_loss
        