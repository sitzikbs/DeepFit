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


class PointNetFeatures(nn.Module):
    def __init__(self, num_points=500, num_scales=1, point_tuple=1, use_point_stn=False, use_feat_stn=False, sym_op='max', concat_prf=False):
        super(PointNetFeatures, self).__init__()
        self.num_points=num_points
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
        self.point_tuple = point_tuple

        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)

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
    def __init__(self, num_points=500, use_point_stn=False, use_feat_stn=False, sym_op='max'):
        super(PointNetEncoder, self).__init__()
        self.pointfeat = PointNetFeatures(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, sym_op=sym_op)
        self.num_points=num_points
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, x):
        n_pts = x.size()[2]
        pointfeat, trans, trans2, trans_points = self.pointfeat(x)

        x = F.relu(self.bn2(self.conv2(pointfeat)))
        x = self.bn3(self.conv3(x))
        global_feature = torch.max(x, 2, keepdim=True)[0]

        return global_feature, pointfeat, trans, trans2, trans_points


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

        pointfeat, trans, trans2 = self.pointfeat(x)
        global_feature = ThreeDmFVNet.get_3DmFV_pytorch(points.permute([0, 2, 1]), self.gmm.weights_, self.gmm.means_,
                                              np.sqrt(self.gmm.covariances_), normalize=True)
        global_feature = torch.flatten(global_feature, start_dim=1)
        x = global_feature.unsqueeze(-1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), global_feature.squeeze(), trans, trans2


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


class Expert(nn.Module):
    def __init__(self, feature_dim=1024, jet_order=1, weight_mode="sigmoid", use_consistancy=False,
                 compute_residuals=False):
        super(Expert, self).__init__()

        self.conv1 = nn.Conv1d(feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, 1, 1) # 1 weight per point
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.jet_order = jet_order
        self.compute_neighbor_normals = use_consistancy
        self.compute_residuals = compute_residuals
        self.weight_mode = weight_mode

    def forward(self, points, x, n_effective_points):
        batchsize = points.size()[0]
        n_points = points.size()[2]

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

        neighbor_normals = None
        residuals = None
        beta, normal, neighbor_normals, residuals = WLSJETFIT(points, weights.squeeze(), order=self.jet_order,
                                                                  compute_neighbor_normals=self.compute_neighbor_normals,
                                                                  compute_residuals=self.compute_residuals)

        return normal, beta.squeeze(), weights.squeeze(),  neighbor_normals, residuals


class ManagerNet(nn.Module):
    def __init__(self, n_experts=3, feature_dim=1024):
        super(ManagerNet, self).__init__()

        self.n_experts = n_experts

        self.conv1 = nn.Conv1d(feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.n_experts, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, global_feature):
        x = F.relu(self.bn1(self.conv1(global_feature)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        expert_probs = F.softmax(F.relu(self.conv4(x)).squeeze(), dim=-1)
        return expert_probs.squeeze()


class DeepFitExperts(nn.Module):
    def __init__(self, num_points=500, use_point_stn=False,  use_feat_stn=False, point_tuple=1,
                 sym_op='max', arch=None, n_gaussians=5, n_experts=2,  weight_mode="sigmoid", use_consistancy=False, compute_residuals=False):
        super(DeepFitExperts, self).__init__()

        self.n_experts = n_experts
        self.n_gaussians = n_gaussians
        self.num_points = num_points
        self.max_N_n = int((self.n_experts + 1) * (self.n_experts + 2) / 2)
        self.arch = arch
        if self.arch == '3dmfv':
            self.n_gaussians = n_gaussians # change later to get this as input
            self.gmm = ThreeDmFVNet.get_3d_grid_gmm(subdivisions=[self.n_gaussians, self.n_gaussians, self.n_gaussians],
                                                    variance=np.sqrt(1.0 / self.n_gaussians))
            self.feature_dim =  self.n_gaussians * self.n_gaussians * self.n_gaussians * 20
            self.global_feature_dim = None #fix later
        else:
            self.pointnetfeat = PointNetEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                        sym_op=sym_op)
            self.global_feature_dim = 1024
            self.feature_dim = self.global_feature_dim + 64

        self.experts = torch.nn.ModuleList()
        for i in range(n_experts):
            self.experts.append(Expert(feature_dim=self.feature_dim, jet_order=i+1, weight_mode=weight_mode,
                                       use_consistancy=use_consistancy, compute_residuals=compute_residuals))

        self.manager = ManagerNet(n_experts=n_experts, feature_dim=self.global_feature_dim)

    def forward(self, x, n_effective_points):

        if self.arch == '3dmfv':
            global_feature = ThreeDmFVNet.get_3DmFV_pytorch(x.permute([0, 2, 1]), self.gmm.weights_, self.gmm.means_,
                                                  np.sqrt(self.gmm.covariances_), normalize=True)
            global_feature = torch.flatten(global_feature, start_dim=1).unsqueeze(-1)
            trans = None
            trans2 = None
        else:
            global_feature, pointfeat, trans, trans2, trans_points = self.pointnetfeat(x)
        pointnet_features = torch.cat([global_feature.view(-1, 1024, 1).repeat(1, 1, x.shape[2]), pointfeat], 1)

        expert_probs = self.manager(global_feature)

        all_expert_normals = []
        all_experts_beta = []
        all_experts_weights = []
        all_experts_neighbor_normals = []
        all_experts_residuals = []


        for i, expert in enumerate(self.experts):
            N_n = int((i+2)*(i+3)/2)
            expert_normal, expert_beta, expert_weight, expert_neighbor_normals, expert_residuals = \
                expert(trans_points, pointnet_features, n_effective_points)
            expert_beta = torch.nn.functional.pad(expert_beta, [0, self.max_N_n - N_n], mode='constant', value=0)
            all_expert_normals.append(expert_normal)
            all_experts_beta.append(expert_beta)
            all_experts_weights.append(expert_weight)

            if expert_neighbor_normals is not None:
                all_experts_neighbor_normals.append(expert_neighbor_normals)
            if expert_residuals is not None:
                all_experts_residuals.append(expert_residuals)

        all_expert_normals = torch.stack(all_expert_normals).permute(1, 0, 2)
        all_experts_beta = torch.stack(all_experts_beta).permute(1, 0, 2)
        all_experts_weights = torch.stack(all_experts_weights).permute(1, 0, 2)

        if expert_neighbor_normals is not None:
            all_experts_neighbor_normals = torch.stack(all_experts_neighbor_normals).permute(1, 0, 2, 3)
        if expert_residuals is not None:
            all_experts_residuals = torch.stack(all_experts_residuals)

        expert_idx = torch.argmax(expert_probs, dim=1)

        n_beta_params = all_experts_beta.shape[2]
        normal = all_expert_normals.gather(1, expert_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)).squeeze()
        beta = all_experts_beta.gather(1, expert_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, n_beta_params)).squeeze()
        weights = all_experts_weights.gather(1, expert_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_points)).squeeze()

        # if expert_neighbor_normals is not None:
        #     all_experts_neighbor_normals = torch.stack(all_experts_neighbor_normals)
        # if expert_residuals is not None:
        #     all_experts_residuals = torch.stack(all_experts_residuals)

        return normal, beta, weights, 0, trans, trans2, expert_probs, all_expert_normals, all_experts_neighbor_normals, all_experts_residuals, all_experts_weights

