# compute normal vectors of a single point cloud
import sys
import os
import numpy as np
sys.path.insert(0, '../utils')
sys.path.insert(0, '../models')
sys.path.insert(0, '../trained_models')
import DeepFit
import tutorial_utils as tu
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./0000000000.xyz', help='full path to input point cloud')
parser.add_argument('--output_path', type=str, default='../log/outputs/', help='full path to input point cloud')
parser.add_argument('--gpu_idx', type=int, default=0, help='index of gpu to use, -1 for cpu')
parser.add_argument('--trained_model_path', type=str, default='../trained_models/DeepFit',
                    help='path to trained model')
parser.add_argument('--mode', type=str, default='classic', help='how to compute normals. use: DeepFit | classic')
parser.add_argument('--k_neighbors', type=int, default=256, help='number of neighboring points for each query point')
parser.add_argument('--jet_order', type=int, default=3,
                    help='order of jet to fit: 1-4. if in DeepFit mode, make sure to match training order')
parser.add_argument('--compute_curvatures', type=bool, default=True, help='true | false indicator to compute curvatures')
args = parser.parse_args()

device = torch.device("cpu" if args.gpu_idx < 0 else "cuda:%d" % 0)

if args.mode == 'DeepFit':
    # load trained model parameters
    params = torch.load(os.path.join(args.trained_model_path, 'DeepFit_params.pth'))
    jet_order = params.jet_order
    print('Using {} order jet for surface fitting'.format(jet_order))
    model = DeepFit.DeepFit(k=1, num_points=args.k_neighbors, use_point_stn=params.use_point_stn,
                            use_feat_stn=params.use_feat_stn, point_tuple=params.point_tuple, sym_op=params.sym_op,
                            arch=params.arch, n_gaussians=params.n_gaussians, jet_order=jet_order,
                            weight_mode=params.weight_mode, use_consistency=False)
    checkpoint = torch.load(os.path.join(args.trained_model_path, 'DeepFit.pth'))
    model.load_state_dict(checkpoint)
    if not (params.points_per_patch == args.k_neighbors):
        print('Warning: You are using a different number of neighbors than trained.')
    model.to(device)
else:
    jet_order = args.jet_order


# load the point cloud
point_cloud_dataset = tu.SinglePointCloudDataset(args.input, points_per_patch=args.k_neighbors)
dataloader = torch.utils.data.DataLoader(point_cloud_dataset, batch_size=128, num_workers=8)


# Estimate normal vectors
for batchind, data in enumerate(dataloader, 0):
    points = data[0]
    data_trans = data[1]
    scale_radius = data[-1]
    points = points.to(device)
    data_trans = data_trans.to(device)
    scale_radius = scale_radius.to(device)

    if args.mode == 'DeepFit':
        # DeepFit
        n_est, beta, weights, trans, trans2, neighbors_n_est = model.forward(points)
        n_est = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)  # cancel out pca
        n_est = n_est.detach().cpu()
        normals = n_est if batchind == 0 else torch.cat([normals, n_est], 0)
    else:
        # Classic, non-weighted jet
        beta, n_est, neighbors_n_est = DeepFit.fit_Wjet(points, torch.ones_like(points[:, 0]), order=jet_order,
                                   compute_neighbor_normals=False)
        n_est = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)  # cancel out pca
        n_est = n_est.detach().cpu()
        normals = n_est if batchind == 0 else torch.cat([normals, n_est], 0)
    # compute principal curvatures
    curv_est, principal_dirs = tu.compute_principal_curvatures(beta)
    curv_est = curv_est / scale_radius.unsqueeze(-1).repeat(1, curv_est.shape[1])
    curv_est = curv_est.detach().cpu()
    curvatures = curv_est if batchind == 0 else torch.cat([curvatures, curv_est], 0)

# Export estimated normals
file_name = os.path.splitext(os.path.basename(args.input))[0]
os.makedirs(args.output_path, exist_ok=True)
normals_output_file_name = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.input))[0] + '.normals')
np.savetxt(normals_output_file_name, normals)
print('Saved estimated normal vectors to ' + normals_output_file_name)

curvatures_output_file_name = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.input))[0] + '.curv')
np.savetxt(curvatures_output_file_name, curvatures.detach().cpu().numpy())
print('Saved estimated principal curvatures to ' + curvatures_output_file_name)