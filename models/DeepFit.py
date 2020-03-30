import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def WLSFIT(points, weights):
    # least squares  fit
    batch_size, D, n_points = points.size()
    x = points[:, :-1, :].squeeze()
    y = points[:, -1, :]
    default_idxs  = torch.arange(len(weights[0]), device=x.device)
    beta_pred = []
    for i in range(batch_size):

        # handle zero weights - if all weights are zero set them to 1
        indx = (weights[i] > 1e-4).nonzero().squeeze()
        if not len(indx.size()) == 0:
            n_valid_points = len(indx)
        else:
            n_valid_points = 0

        if n_valid_points > 3:
            W = torch.diag(weights[i, indx].squeeze())
        else:
            n_valid_points = n_points
            W = torch.diag(torch.ones_like(weights[i]))
            indx = default_idxs

        A = torch.cat([x[i, :, indx], torch.ones(1, n_valid_points, dtype=x.dtype, device=x.device)], dim=0).t()
        XtWX = torch.mm(A.t(), torch.mm(W, A))
        XtY = torch.mm(A.t(), torch.mm(W, y[i, indx].unsqueeze(-1)))
        L = torch.cholesky(XtWX)

        beta = torch.cholesky_solve(XtY, L)
        beta_pred.append(beta)
    beta_pred = torch.stack(beta_pred).squeeze()

    return beta_pred

class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1)


class SimpPointNet(nn.Module):
    def __init__(self, k=2):
        super(SimpPointNet, self).__init__()
        self.k = k  # k is the number of weights per point e.g. 1
        self.feat = PointNetEncoder()
        self.conv1 = nn.Conv1d(1024 + 64, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, points):
        # batchsize = x.size()[0]
        # n_pts = x.size()[2]
        x = self.feat(points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        weights = torch.ones_like(x) + x  # learn the residual->weights start at 1
        x = WLSFIT(points, weights.squeeze())
        return x.squeeze(), weights.squeeze() #beta, weights


