from utils import intersection_over_union
import torch
import torch.nn as nn

class yololoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(yololoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5