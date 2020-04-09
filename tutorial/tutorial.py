import sys
sys.path.insert(0, '../utils')
sys.path.insert(0, '../models')
sys.path.insert(0, '../trained_models')
import DeepFit
import tutorial_utils as tu
import torch

# Generate data - Synthetic example
jet_order = 1
n_points = 4096
point_cloud_dataset = tu.SyntheticPointCloudDataset(n_points, jet_order,points_per_patch=128)
# dataloader = torch.utils.data.DataLoader(point_cloud_dataset, batch_size=256, num_workers=8)
#
# jet_order_fit = 3
# for batchind, points in enumerate(dataloader):
#     pca_beta, n_est, _ = DeepFit.fit_Wjet(points, torch.ones_like(points[:, 0]), order=1,
#                                           compute_neighbor_normals=False)
#     pca_normals = n_est if batchind == 0 else torch.cat([pca_normals, n_est], 0)
#
#     jet_beta, n_est, _ = DeepFit.fit_Wjet(points, torch.ones_like(points[:, 0]), order=jet_order_fit,
#                                           compute_neighbor_normals=False)
#     jet_normals = n_est if batchind == 0 else torch.cat([jet_normals, n_est], 0)
#
# n_sign = torch.sign(torch.sum(pca_normals*
#                               torch.tensor([0., 0., 1.]).repeat([n_points, 1]), dim=1)).unsqueeze(-1)
# pca_normals = n_sign * pca_normals
# pca_normals = pca_normals.detach().cpu().numpy()