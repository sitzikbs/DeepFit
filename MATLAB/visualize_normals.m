clear all
close all
clc
data_path = '../log/outputs/';
normals_path = '../log/outputs/';
file_name = '0000000000';
pc_filename = [data_path, file_name, '.xyz'];
normals_filename = [normals_path, file_name, '.normals'];
output_ply_filename = [normals_path, file_name, '.ply'];
points= dlmread(pc_filename);
normals = dlmread(normals_filename);

fig_h = figure();
ax_h = axes('position',[0, 0, 1, 1]);
set_vis_props(fig_h, ax_h);
pc_h = scatter3(points(:, 1), points(:, 2), points(:, 3), 100, '.');
axis off
mapped_normal_colors = Sphere2RGBCube(sign(sum(points.*normals,2)).*normals);
pc_h.CData = mapped_normal_colors;