clear all
close all
clc
data_path = '/home/sitzikbs/Datasets/pcpnet/';
file_sets_path = '/home/sitzikbs/Datasets/pcpnet/'; % directory containing lists of files ,camera parameters and shape list
k_neighbors = 256;
jet_order = 3;
results_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/jetnet_nci_new3/ablations/Deepfit_knn_lr0.001_sigmoid_cr_log_d3_p256_Lsin/results/';
output_path = [results_path, 'images/teaser/'] ;
    if ~exist(output_path,'dir')
        mkdir(output_path)
    end
    
file_list_file_name = 'vis_set.txt';
shape_list_file_name = 'testset_no_noise.txt';
point_size = 1;

cam_params = camera_parameters();
xyz_file_list = dir([data_path,'*.xyz']);
shapes_list = [file_sets_path, shape_list_file_name];
file_list_to_export = [file_sets_path, file_list_file_name];
shapes_list = strsplit(fileread(shapes_list));
shapes_list = shapes_list(~cellfun('isempty',shapes_list));  % remove empty cells
shapes_to_export = strsplit(fileread(file_list_to_export));
shapes_to_export = shapes_to_export(~cellfun('isempty',shapes_to_export));  % remove empty cells

shapes_to_export = {'Liberty100k'};
scale_list = zeros(size(shapes_list));
mean_list = zeros(size(shapes_list,2), 3);

for shape = shapes_to_export

    xyz_file_name = [data_path, shape{1}, '.xyz'];
    normals_gt_filename = [data_path, shape{1}, '.normals'];
    curvs_gt_filename = [data_path, shape{1}, '.curv'];
    normals_filename = [results_path, shape{1}, '.normals'];
    weights_filename = [results_path, shape{1}, '.weights'];
    beta_filename = [results_path, shape{1}, '.beta'];
    trans_filename = [results_path, shape{1}, '.trans'];
    original_points = dlmread(xyz_file_name);
    normals = dlmread(normals_filename);
    normals_gt = dlmread(normals_gt_filename);
    curvs_gt = dlmread(curvs_gt_filename);
    weights = dlmread(weights_filename);
    beta = dlmread(beta_filename);
    trans = dlmread(trans_filename);
%     trans = reshape(trans, [size(trans,1),3,3]);

    fig_h = figure();
    ax_h = axes('position',[0, 0, 1, 1]);
    set_vis_props(fig_h, ax_h);

    xlim([-1, 1]);
    ylim([-1, 1]);
    zlim([-1, 1]);
    
    for i= 1:size(shapes_list, 2) 
        if contains(shape, shapes_list{i})
            shape_idx = i;
        end
    end
    
    points = original_points;
    if scale_list(shape_idx) == 0 
         mean_list(shape_idx, :) = mean(points);
         points = points - mean_list(shape_idx, :); 
         scale = norm(max(points) - min(points)); %bbox
         scale_list(shape_idx) = 1 ./ scale;
         points = points.*scale_list(shape_idx); 
    else
         points = points - mean_list(shape_idx, :); 
         points = points.*scale_list(shape_idx); 
    end 
    
    ptCloud = pointCloud(original_points);
    
    querypoint_indx = 65306; %61433; %83935; %36527; % 65306;
    [nn_indices, nn_dist] = findNearestNeighbors(ptCloud, original_points(querypoint_indx, :), k_neighbors, 'Sort', true);
    % find closest point
%   [closest_ind, ~] = findNearestNeighbors(ptCloud,[0.08131, 0.3345, -0.01635],1);
    
    pc_h = scatter3(points(:, 1), points(:, 2), points(:, 3), point_size, '.');
    hold all
    scatter3(points(querypoint_indx, 1), points(querypoint_indx, 2), points(querypoint_indx, 3), point_size, '.', 'k');
%     points(:, 2) = -points(:, 2);
    if contains(shape, cam_params.name)
        indx=find(cellfun(@contains,repmat({shape},size(cam_params.name)),cam_params.name)==1);
        Reshape(gca, cam_params.r{indx}, cam_params.phi{indx}, cam_params.theta{indx}, [0, 0, 0], cam_params.view_angle{indx}, cam_params.permute{indx});
%         points(:, 1) = cam_params.x_scale{indx} * points(:, 1); %flip y axis for color coding
%         points(:, 2) = cam_params.y_scale{indx} * points(:, 2); %flip y axis for color coding
%         points(:, 3) = cam_params.z_scale{indx} * points(:, 3); %flip y axis for color coding
    else
        Reshape(gca, 10, 45, 35.26, [0, 0, 0], 20, [1, 2, 3]);
    end
    

    pc_h.CData = -points(:, 2);
    cmin = min(pc_h.CData);
    cmax = max(pc_h.CData);
    cmap = colormap;
    m = length(cmap);
    index = fix((pc_h.CData-cmin)/(cmax-cmin)*m)+1; %A
    % Then to RGB
    RGB = squeeze(ind2rgb(index,cmap));
    pc_h.CData = RGB;
    % save full point cloud
    image_filename = [output_path, shape{1}, '.png'];
    print(image_filename, '-dpng')
    
    % save full point cloud with neighbors highlghted
    pc_h.CData(nn_indices, :) = repmat([1, 0, 0], [k_neighbors, 1]);
    % save full point cloud
    image_filename = [output_path, shape{1}, '_highlighted.png'];
    print(image_filename, '-dpng')
    
    %save normal visualization
    pc_h.CData = Sphere2RGBCube(normals_gt);
    image_filename = [output_path, shape{1}, 'normals.png'];
    print(image_filename, '-dpng')
    
    %save curvature visualization
    k1rang = [mean(curvs_gt(:, 1)) - 2*std(curvs_gt(:, 1)), mean(curvs_gt(:, 1)) + 2*std(curvs_gt(:, 1))];
    k2rang = [mean(curvs_gt(:, 2)) - 2*std(curvs_gt(:, 2)), mean(curvs_gt(:, 2)) + 2*std(curvs_gt(:, 2))];
    pc_h.CData = curvature_color_mapping(curvs_gt, k1rang, k2rang);%[min(curvs_gt(:, 1)), max(curvs_gt(:, 1))], [min(curvs_gt(:, 2)), max(curvs_gt(:, 2))]);
    image_filename = [output_path, shape{1}, 'curvs_map.png'];
    print(image_filename, '-dpng')
    
    pc_h.CData = mean(curvs_gt, 2);
    colormap jet
    caxis([mean(mean(curvs_gt, 2))-3*std(mean(curvs_gt, 2)), mean(mean(curvs_gt, 2))+3*std(mean(curvs_gt, 2))]);
    image_filename = [output_path, shape{1}, 'curvs_avg.png'];
    print(image_filename, '-dpng')
    colormap parula
    %save local cloud with original colormap
    figure();
    axes();
    original_local_points = original_points(nn_indices, :);
    [local_points, data_trans, scale] = align_local_cloud(original_local_points, original_points, nn_dist);
    normals_gt(querypoint_indx, :) = normals_gt(querypoint_indx, :) * data_trans;
    normals(querypoint_indx, :) = normals(querypoint_indx, :) * data_trans;
    pc_local_h = scatter3(local_points(:, 1), local_points(:, 2), local_points(:, 3), 600, '.');
    hold all
%     quiver3(local_points(1, 1), local_points(1, 2), local_points(1, 3), normals_gt(querypoint_indx, 1), normals_gt(querypoint_indx, 2), normals_gt(querypoint_indx, 3), 0.3, 'g');
%     quiver3(local_points(1, 1), local_points(1, 2), local_points(1, 3), normals(querypoint_indx, 1), normals(querypoint_indx, 2), normals(querypoint_indx, 3), 0.3, 'r');
    xlim([-1, 1]);
    ylim([-1, 1]);
    zlim([-1, 1]);
    pc_local_h.CData = -local_points(:, 2);
    set_vis_props(fig_h, ax_h);
    image_filename = [output_path, shape{1}, '_local_original.png'];
    print(image_filename, '-dpng')
    
    % save weight visualiation
    pc_local_h.CData = weights(querypoint_indx, :)';
%     caxis([min(weights(querypoint_indx, :)), mean(weights(querypoint_indx, :) + std(weights(querypoint_indx, :)))]);
    caxis([0, 1])
    image_filename = [output_path, shape{1}, '_weights.png'];
    print(image_filename, '-dpng')
    
    %save jet visualization
    orientation_sign = -sign(sum(normals(querypoint_indx, :) .* normals_gt(querypoint_indx, :)));
    beta(querypoint_indx, 3:end) = beta(querypoint_indx, 3:end) * orientation_sign;
    surf_h = plot_jet(orientation_sign*beta(querypoint_indx, :), jet_order, local_points, reshape(trans(querypoint_indx, :),3,3));
    light('position', [0, 0, 1]);
    material shiny
    lighting gouraud
    
%     surf_h.FaceAlpha = 1;
%     surf_h.EdgeColor = 'r';
%     surf_h.EdgeAlpha = 0.5;
    
    surf_h.EdgeColor = [0.7, 0, 0];
    surf_h.EdgeAlpha = 0.7;
    surf_h.FaceAlpha = 0.7;


    image_filename = [output_path, shape{1}, '_jet.png'];
    print(image_filename, '-dpng')
end

function [points, U, scale]= align_local_cloud(points, global_points, nn_dist)

% bbdiag = norm(max(global_points) - min(global_points));
% scale = 0.05 * bbdiag;
scale = max(nn_dist);

points = points - points(1, :);
points = points / scale; 

% pts_mean = mean(points, 1);
% points = points - pts_mean;

[U, S, V] = svd(points');
% cov_mat = (1/npoints)* points' * points;
% [U,D] = eig(cov_mat);

points = points*U;
% cp_new = -pts_mean;
% cp_new = cp_new*U;
% points = points - cp_new;

% points = points';
end