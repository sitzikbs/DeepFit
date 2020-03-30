import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import normal_estimation_utils
import ThreeDmFVNet
from DeepFitJet import fit_Wjet as WLSJETFIT

def WLSFIT(points, weights, n_effective_points, normalize=True):
    # least squares  fit
    batch_size, D, n_points = points.size()
    beta_pred = []
    n_pred = []
    for i in range(batch_size):
        x = points[i, 0, 0:n_effective_points[i]]
        y = points[i, 1, 0:n_effective_points[i]]
        z = points[i, 2, 0:n_effective_points[i]]
        effective_weights = weights[i, 0:n_effective_points[i]]
        default_idxs = torch.arange(n_effective_points[i], device=x.device, dtype=torch.long)

        # handle zero weights - if all weights are zero set them to 1
        indx = (effective_weights > 1e-4).nonzero().squeeze()
        if not len(indx.size()) == 0:
            n_valid_points = len(indx)
        else:
            n_valid_points = 0

        if n_valid_points > 3:
            x = x[indx]
            y = y[indx]
            z = z[indx]
            W = torch.diag(effective_weights[indx].squeeze())
            w = effective_weights[indx].squeeze()
        else:
            n_valid_points = n_effective_points[i]
            W = torch.diag(torch.ones_like(effective_weights))
            w = torch.ones_like(effective_weights, requires_grad=True)
            indx = default_idxs

        wx, wy, wz = w * x,  w * y,  w * z
        cg_x, cg_y, cg_z = torch.mean(wx), torch.mean(wy), torch.mean(wz)
        wx, wy, wz = wx - cg_x,  wy - cg_y,  wz - cg_z
        xx, yy, zz = torch.sum(wx * wx), torch.sum(wy * wy), torch.sum(wz * wz)
        xy, xz, yz = torch.sum(wx * wy), torch.sum(wx * wz), torch.sum(wy * wz)
        Dx, Dy, Dz = yy * zz - yz * yz, xx * zz - xz * xz, xx * yy - xy * xy
        max_det = torch.max(torch.stack([Dx, Dy, Dz]))
        if max_det == Dz:
            nx, ny, nz = xy * yz - xz * yy, xy * xz - yz * xx,  Dz
        elif max_det == Dy:
            nx, ny, nz = xz * yz - xy * zz, Dy, xy * xz - yz * xx
        else:  # max_det == Dx
            nx, ny, nz = Dx, xz * yz - xy * zz, xy * yz - xz * yy

        n = torch.stack([nx, ny, nz])
        if normalize:
            n = torch.nn.functional.normalize(n, p=2, dim=0)
        beta = torch.stack([nx, ny, nz, nx * cg_x + ny * cg_y + nz * cg_z])

        beta_pred.append(beta)
        n_pred.append(n)
    beta_pred = torch.stack(beta_pred).squeeze()
    n_pred = torch.stack(n_pred).squeeze()

    return beta_pred, n_pred


def fit_jet(points, W, order=2, xyz_perm=[0, 1, 2]):
    # compute the vandermonde matrix
    x = points[xyz_perm[0]].unsqueeze(1)
    y = points[xyz_perm[1]].unsqueeze(1)
    z = points[xyz_perm[2]].unsqueeze(1)
    if order == 1:
        A = torch.cat([x, y, torch.ones_like(x)], dim=1)
    elif order == 2:
        A = torch.cat([x, y, x * x, y * y, x * y, torch.ones_like(x)], dim=1)
    else:
        raise ValueError("Polynomial order unsupported, please use 1 or 2 ")

    XtWX = torch.mm(A.t(), torch.mm(W, A))
    XtY = torch.mm(A.t(), torch.mm(W, z))
    L = torch.cholesky(XtWX)
    beta_perm = torch.cholesky_solve(XtY, L)

    return beta_perm.squeeze()

class PointNetFeatures(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', concat_prf=False):
        super(PointNetFeatures, self).__init__()
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales
        if not concat_prf:
            self.conv1 = torch.nn.Conv1d(3, 64, 1)
        else:
            self.conv1 = torch.nn.Conv1d(5, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.concat_prf = concat_prf


        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

    def forward(self, x):
        n_pts = x.size()[2]
        points = x
        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3 * self.point_tuple, -1)
            points = x
        else:
            trans = None

        if self.concat_prf:
            x = torch.cat([x, torch.abs(x[:, 2, :]).unsqueeze(1), torch.sum(torch.pow(x, 2), dim=1).unsqueeze(1)], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        return x,  trans, trans2, points


class PointNetEncoder(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', concat_prf=False):
        super(PointNetEncoder, self).__init__()
        self.pointfeat = PointNetFeatures(num_points=num_points, num_scales=num_scales, use_point_stn=use_point_stn,
                         use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op, concat_prf=concat_prf)
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, points):
        n_pts = points.size()[2]
        pointfeat, trans, trans2, points = self.pointfeat(points)

        x = F.relu(self.bn2(self.conv2(pointfeat)))
        x = self.bn3(self.conv3(x))
        global_feature = torch.max(x, 2, keepdim=True)[0]
        x = global_feature.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), global_feature.squeeze(), trans, trans2, points


class PointNet3DmFVEncoder(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', n_gaussians=5):
        super(PointNet3DmFVEncoder, self).__init__()
        self.num_points = num_points
        self.point_tuple = point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales = num_scales
        self.pointfeat = PointNetFeatures(num_points=num_points, num_scales=num_scales, use_point_stn=use_point_stn,
                         use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op)

        self.n_gaussians = n_gaussians

        self.gmm = ThreeDmFVNet.get_3d_grid_gmm(subdivisions=[self.n_gaussians, self.n_gaussians, self.n_gaussians],
                              variance=np.sqrt(1.0 / self.n_gaussians))


    def forward(self, x):
        points = x
        n_pts = x.size()[2]

        pointfeat, trans, trans2, points = self.pointfeat(points)
        global_feature = ThreeDmFVNet.get_3DmFV_pytorch(points.permute([0, 2, 1]), self.gmm.weights_, self.gmm.means_,
                                              np.sqrt(self.gmm.covariances_), normalize=True)
        global_feature = torch.flatten(global_feature, start_dim=1)
        x = global_feature.unsqueeze(-1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), global_feature.squeeze(), trans, trans2, points


class ResPointNet(nn.Module):
    def __init__(self, k=2, num_points=500, fit_type='plane', use_point_stn=False,  use_feat_stn=False, point_tuple=1, sym_op='max'):
        super(ResPointNet, self).__init__()
        self.k = k  # k is the number of weights per point e.g. 1
        self.fit_type = fit_type
        self.feat = PointNetEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op)
        self.conv1 = nn.Conv1d(1024 + 64, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.3)
        self.do2 = nn.Dropout(p=0.3)

    def forward(self, points, n_effective_points):
        # batchsize = x.size()[0]
        # n_pts = x.size()[2]
        x, _, trans, trans2 = self.feat(points)
        xw = F.relu(self.bn1(self.conv1(x)))
        xw = F.relu(self.bn2(self.conv2(xw)))
        xw = F.relu(self.bn3(self.conv3(xw)))
        xw = torch.tanh(self.conv4(xw))

        weights = torch.ones_like(xw) + xw  # learn the residual->weights start at 1
        if self.fit_type == 'plane':
            x, normal = WLSFIT(points, weights.squeeze(), n_effective_points, normalize=True)
        elif  self.fit_type == 'jet':
            x, normal = WLSJETFIT(points, weights.squeeze(), n_effective_points)

        # estimate and add residual
        _, global_feature = self.feat(points)
        x_res = F.relu(self.bn1(self.fc1(global_feature)))
        x_res = self.do1(x_res)
        x_res = F.relu(self.bn2(self.fc2(x_res)))
        x_res = self.do2(x_res)
        x_res = self.fc3(x_res)
        normal = normal + x_res
        normal = torch.nn.functional.normalize(normal, p=2, dim=0)
        return normal, x.squeeze(), weights.squeeze(), x_res, trans, trans2  # normal, beta, weights


class SimpPointNet(nn.Module):
    def __init__(self, k=2, num_points=500, fit_type='plane', use_point_stn=False,  use_feat_stn=False, point_tuple=1,
                 sym_op='max', arch=None, n_gaussians=5, jet_order=2, concat_prf=False, weight_mode="tanh",
                 use_consistency=False, compute_residuals=False):
        super(SimpPointNet, self).__init__()
        self.k = k  # k is the number of weights per point e.g. 1
        self.num_points=num_points
        self.point_tuple = point_tuple
        if arch == '3dmfv':
            self.n_gaussians = n_gaussians  # change later to get this as input
            self.feat = PointNet3DmFVEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                        point_tuple=point_tuple, sym_op=sym_op, n_gaussians= self.n_gaussians )
            feature_dim = self.n_gaussians * self.n_gaussians * self.n_gaussians * 20 + 64
        else:
            self.feat = PointNetEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                            point_tuple=point_tuple, sym_op=sym_op, concat_prf=concat_prf)

            feature_dim = 1024 + 64
        self.conv1 = nn.Conv1d(feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.fit_type = fit_type
        self.jet_order = jet_order
        self.weight_mode = weight_mode
        self.compute_neighbor_normals = use_consistency
        self.compute_residuals = compute_residuals
        self.do = torch.nn.Dropout(0.25)
        if weight_mode == "variational":
            self.normal_dist = torch.distributions.normal.Normal(torch.tensor(0.0).to(device=torch.device("cuda")),
                                                                 torch.tensor(1.0).to(device=torch.device("cuda")),
                                                                 validate_args=None)
            self.conv5 = nn.Conv1d(128, 2, 1)

    def forward(self, points, n_effective_points):
        batchsize = points.size()[0]
        n_points = points.size()[2]
        x, _, trans, trans2, points = self.feat(points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # point weight estimation. consider adding dropout
        if self.weight_mode == "softmax":
            x = F.softmax(self.conv4(x))
            weights = 0.01 + x  # add epsilon for numerical robustness
        elif self.weight_mode =="tanh":
            x = torch.tanh(self.conv4(x))
            weights = (0.01 + torch.ones_like(x) + x) / 2.0  # learn the residual->weights start at 1
        elif self.weight_mode =="sigmoid":
            weights = 0.01 + torch.sigmoid(self.conv4(x))
        elif self.weight_mode =="variational":
            sample = self.normal_dist.sample([batchsize, n_points])
            dist_params = torch.sigmoid(self.conv5(x))
            weights = dist_params[:, 0] + dist_params[:, 1] * sample
            weights[weights < 0.0] = 0.01
            weights[weights > 1.0] = 1.0

        # weights = self.do(weights)  # drop out
        neighbor_normals = None
        residuals = None
        # weights = torch.ones_like(x, requires_grad=True)  # debugging

        if self.fit_type == 'plane':
            beta, normal = WLSFIT(points, weights.squeeze(), n_effective_points)
        elif self.fit_type == 'jet':
            # x, normal = WLSJETFIT(points, weights.squeeze(), n_effective_points)
            beta, normal, neighbor_normals, residuals = WLSJETFIT(points, weights.squeeze(), order=self.jet_order,
                                                                  compute_neighbor_normals=self.compute_neighbor_normals,
                                                                  compute_residuals=self.compute_residuals)

        return normal, beta.squeeze(), weights.squeeze(), 0, trans, trans2, neighbor_normals, residuals


class VarPointNet(nn.Module):
    def __init__(self, k=2, num_points=500, fit_type='plane', use_point_stn=False,  use_feat_stn=False, point_tuple=1,
                 sym_op='max', arch=None, n_gaussians=5, jet_order=2, concat_prf=False, weight_mode="tanh",
                 use_consistency=False, compute_residuals=False):
        super(VarPointNet, self).__init__()
        self.k = k  # k is the number of weights per point e.g. 1
        self.num_points=num_points
        self.point_tuple = point_tuple
        if arch == '3dmfv':
            self.n_gaussians = n_gaussians  # change later to get this as input
            self.feat = PointNet3DmFVEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                        point_tuple=point_tuple, sym_op=sym_op, n_gaussians= self.n_gaussians )
            feature_dim = self.n_gaussians * self.n_gaussians * self.n_gaussians * 20 + 64
        else:
            self.feat = PointNetEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                            point_tuple=point_tuple, sym_op=sym_op, concat_prf=concat_prf)

            feature_dim = 1024 + 64
        self.conv1 = nn.Conv1d(feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.fit_type = fit_type
        self.jet_order = jet_order
        self.weight_mode = weight_mode
        self.compute_neighbor_normals = use_consistency
        self.compute_residuals = compute_residuals
        self.do = torch.nn.Dropout(0.25)
        # self.normal_dist = torch.distributions.normal.Normal(torch.tensor(0.0).to(device=torch.device("cuda")),
        #                                                      torch.tensor(1.0).to(device=torch.device("cuda")),
        #                                                      validate_args=None)
        self.conv4 = nn.Conv1d(128, 1, 1)
        self.conv5 = nn.Conv1d(128, 1, 1)

    def forward(self, points, n_effective_points):
        batchsize = points.size()[0]
        n_points = points.size()[2]
        x, global_feature, trans, trans2, points = self.feat(points)
        # distribution_params = torch.relu(self.fc1(global_feature))
        # distribution_params = torch.sigmoid(self.fc2(distribution_params))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # point weight estimation. consider adding dropout
        # if self.weight_mode == "softmax":
        #     x = F.softmax(self.conv4(x) / torch.sqrt(torch.tensor(n_points, device=points.device, dtype=points.dtype)), dim=2)
        #     weights = 0.01 + x  # add epsilon for numerical robustness
        # elif self.weight_mode =="tanh":
        #     x = torch.tanh(self.conv4(x))
        #     weights = (0.01 + torch.ones_like(x) + x) / 2.0  # learn the residual->weights start at 1
        # elif self.weight_mode =="sigmoid":
        #     weights = 0.01 + torch.sigmoid(self.conv4(x))
        # elif self.weight_mode =="variational":
        # sample = self.normal_dist.sample([batchsize, n_points])
        mu = torch.sigmoid(self.conv5(x))
        logvar = self.conv4(x)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        weights = mu + eps * std
        weights[weights <= 0.0] = 0.01
        weights[weights > 1.0] = 1.0
        distribution_params  = torch.cat([mu, logvar], dim=1)
        # weights = self.do(weights)  # drop out
        neighbor_normals = None
        residuals = None
        # weights = torch.ones_like(x, requires_grad=True)  # debugging

        if self.fit_type == 'plane':
            beta, normal = WLSFIT(points, weights.squeeze(), n_effective_points)
        elif self.fit_type == 'jet':
            # x, normal = WLSJETFIT(points, weights.squeeze(), n_effective_points)
            beta, normal, neighbor_normals, residuals = WLSJETFIT(points, weights.squeeze(), order=self.jet_order,
                                                                  compute_neighbor_normals=self.compute_neighbor_normals,
                                                                  compute_residuals=self.compute_residuals)

        return normal, beta.squeeze(), weights.squeeze(), 0, trans, trans2, neighbor_normals, residuals, distribution_params


class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


class QSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = normal_estimation_utils.batch_quat_to_rotmat(x)

        return x