
clear all
close all
clc
data_path = '/home/sitzikbs/Datasets/pcpnet/';
% normals_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/jetnet_nci_new3/ablations/Deepfit_knn_lr0.001_sigmoid_cr_log_d3_p256_Lsin/results/';
normals_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/excluded_baselines/jet/112_neighbors/';
file_name = 'boxunion2100k';
pc_filename = [data_path, file_name, '.xyz'];
normals_gt_filename = [data_path, file_name, '.normals'];
normals_filename = [normals_path, file_name, '.normals'];
ply_filename = [normals_path, file_name, '.ply'];
points= dlmread(pc_filename);
normals = dlmread(normals_filename);
normals_gt = dlmread(normals_gt_filename);
normal_orientation = sign(sum(normals'.*normals_gt'))';
normals = normals.*(repmat(normal_orientation, 1,3));
pcCloud_obj = pointCloud(points, 'Normal', normals);
pcwrite(pcCloud_obj,ply_filename)