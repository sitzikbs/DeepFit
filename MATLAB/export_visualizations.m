function export_visualizations()
clear all
close all
clc

data_path = '/home/sitzikbs/Datasets/pcpnet/';
file_sets_path = '/home/sitzikbs/Datasets/pcpnet/'; % directory containing lists of files ,camera parameters and shape list

results_path = '/mnt/3.5TB_WD/PycharmProjects/SeNSor/log/v0/results/';
% results_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/excluded_baselines/jet/112_neighbors/';
% results_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/cgal_cpp/results/cgal/jet_large/';
% results_path = '/mnt/sitzikbs_storage/PycharmProjects/pcpnet/results/PCPNet_ms_normal_curv/';
file_list_file_name = 'testset_no_noise.txt';
shape_list_file_name = 'testset_no_noise.txt';

output_path = [results_path, 'images/'] ;
    if ~exist(output_path,'dir')
        mkdir(output_path)
    end
    
export_type = 'all'; %shape name for single point cloud export or all for all point clouds

export_point_vis = true;
export_normal_vis = true;
% export_expert_vis = false;
export_normal_error_vis = true;
export_curvature_vis = false;
export_curvature_error_vis = false;
curvature_type = 'both';
feature_type = 'curvatures';
curv_err_type = 'L';
max_err_ang = 60;

use_subset = false;

if use_subset
    point_size = 400;
else
     point_size = 10;
end

% check_path(true, output_path);

% setup all output directories
[xyz_img_output_path, normals_img_output_path, ...
    normal_error_img_output_path, curvatures_img_output_path, ...
    curvatures_error_img_output_path] = output_directory_setup(...
    output_path, export_point_vis, export_normal_vis,...
    export_normal_error_vis, export_curvature_vis,...
    export_curvature_error_vis);


cam_params = camera_parameters();
xyz_file_list = dir([data_path,'*.xyz']);
% cam_params_file =  [file_sets_path, 'camera_parameters.txt'];
shapes_list = [file_sets_path, shape_list_file_name];
file_list_to_export = [file_sets_path, file_list_file_name];
shapes_list = strsplit(fileread(shapes_list));
shapes_list = shapes_list(~cellfun('isempty',shapes_list));  % remove empty cells
shapes_to_export = strsplit(fileread(file_list_to_export));
shapes_to_export = shapes_to_export(~cellfun('isempty',shapes_to_export));  % remove empty cells
scale_list = zeros(size(shapes_list));
mean_list = zeros(size(shapes_list,2), 3);
if ~strcmp(export_type, 'all')
    shapes_to_export = shapes_to_export(contains( shapes_to_export, export_type));
end
% cam_params = dlmread(cam_params_file);

for shape = shapes_to_export
    disp(['saving ', shape{1}, '...']);
    xyz_file_name = [data_path, shape{1}, '.xyz'];
    normals_gt_file_name = [data_path, shape{1}, '.normals'];
    normals_file_name =  [results_path, shape{1}, '.normals'];
%     expert_file_name = [results_path, shape{1}, '.experts'];
    idx_file_name =  [data_path, shape{1}, '.pidx'];
    curvatures_gt_file_name = [data_path, shape{1}, '.curv'];
    curvatures_file_name = [results_path, shape{1}, '.curv'];
    points = dlmread(xyz_file_name);
    
    for i= 1:size(shapes_list, 2) 
        if contains(shape, shapes_list{i})
            shape_idx = i;
        end
    end
    
    if scale_list(shape_idx) == 0 
         mean_list(shape_idx, :) = mean(points);
         points = points - mean_list(shape_idx, :); 
         scale_list(shape_idx) = (1./max(sqrt(sum(points.^2, 2))));
         points = points.*scale_list(shape_idx); 
    else
         points = points - mean_list(shape_idx, :); 
         points = points.*scale_list(shape_idx); 
    end
    normals_gt = dlmread(normals_gt_file_name);
    curvatures_gt = dlmread(curvatures_gt_file_name); 
    
    % check if estimations were performed on subset or full set
    if export_normal_vis
        normals = dlmread(normals_file_name);
        n_features = size(normals, 1);
    end
    if export_curvature_vis
        curvatures = dlmread(curvatures_file_name); 
        curvatures = curvatures(:, 1:2);
        n_features = size(curvatures, 1);
        orientation_sign = sign(sum((normals.* normals_gt)'));
        curvatures = repmat(orientation_sign', 1, 2).* curvatures;
        curvatures = curvatures';
        [~,index] = max(abs(curvatures));
        absmax = curvatures(sub2ind(size(curvatures),index,1:size(curvatures,2)));
        [~,index] = min(abs(curvatures));
        absmin = curvatures(sub2ind(size(curvatures),index,1:size(curvatures,2)));
        curvatures = [absmax', absmin'];
        % map curvatures
    end
    
    npoints = size(points,1); % new number of points
    if npoints ~= 5000
%         idxs = dlmread(idx_file_name) + 1;
%         points = points(idxs, :);
%         if strcmp(feature_type, 'curvatures')
%             curvatures_gt = curvatures_gt(idxs,:);
%         else
%             normals_gt = normals_gt(idxs, :);
%         end
    else
        idxs = dlmread(idx_file_name) + 1;
        points = points(idxs, :);
        if strcmp(feature_type, 'curvatures')
            curvatures_gt = curvatures_gt(idxs,:);
            curvatures = curvatures(idxs,:);
        else
            normals_gt = normals_gt(idxs, :);
            normals = normals(idxs, :);
        end  
        expert = expert(idxs, :);
    end
    npoints = size(points,1); % new number of points
    visiblePtInds = 1:npoints;
    
    fig_h = figure();
    ax_h = axes('position',[0, 0, 1, 1]);
    set_vis_props(fig_h, ax_h);
%     if ~strcmp(shapes_list{shape_idx}, 'Cup34100k')
%      	Reshape(gca, cam_params(shape_idx, 1), cam_params(shape_idx, 2), cam_params(shape_idx, 3));
%     else
%         Reshape_cup(gca, cam_params(shape_idx, 1), cam_params(shape_idx, 2), cam_params(shape_idx, 3));
%     end
%     Reshape(gca, 5, 45+180, 35.26);
        if contains(shape, cam_params.name)
            indx=find(cellfun(@contains,repmat({shape},size(cam_params.name)),cam_params.name)==1);
            Reshape(gca, cam_params.r{indx}, cam_params.phi{indx}, cam_params.theta{indx}, [0, 0, 0], cam_params.view_angle{indx}, cam_params.permute{indx});
        else
            Reshape(gca, 10, 45, 35.26, [0, 0, 0], 15, [1, 2, 3]);
        end
        
    xlim([-1, 1]);
    ylim([-1, 1]);
    zlim([-1, 1]);

    points = points (visiblePtInds,:);
    normals_gt = normals_gt(visiblePtInds,:);
    normals = normals(visiblePtInds,:);
    
    pc_h = scatter3(points(:, 1), points(:, 2), points(:, 3), point_size, '.');
    axis off
    if export_point_vis
        image_filename = [xyz_img_output_path, shape{1}, '.png'];
        print(image_filename, '-dpng')
    end 
    
    if export_normal_vis
        mapped_normal_colors = Sphere2RGBCube(sign(sum(normals_gt.*normals,2)).*normals);
        pc_h.CData = mapped_normal_colors;
        image_filename = [normals_img_output_path, shape{1}, '.png'];
        print(image_filename, '-dpng')
    end
    
    if export_normal_error_vis     
%         error = min(sqrt(sum((normals_gt - normals).^2, 2)), sqrt(sum((normals_gt + normals).^2, 2))); % consider visualizing the sin/ cosine error
        diff = abs(sum(normals.*normals_gt,2))./ (sqrt(sum(normals.^2,2)).* sqrt(sum(normals_gt.^2,2)));
        diff(diff > 1) = 1;
        error = acosd(diff);
        rms = sqrt(mean(error.^2));
        colormap('parula');
        caxis([0, max_err_ang]);
        pc_h.CData = error;
        active_axes = gca;
        active_axes.Position =  [0, 0, 1, 0.9];
        ax_rns_text = axes('position',[0, 0.9, 1, 0.1]);
        axis off
        text('string',num2str(rms), 'position',[0.5, 0.5], 'FontSize', 24, 'HorizontalAlignment', 'center')
        image_filename = [normal_error_img_output_path, shape{1}, '.png'];    
        print(image_filename, '-dpng')
        gcf.CurrentAxes  = active_axes;
        active_axes.Position =  [0, 0, 1, 1];
        delete(ax_rns_text)
    end
    
    if export_curvature_vis
        normal_sign = sign(sum(normals_gt.*normals, 2));
%         curvatures = repmat(normal_sign, [1, 2]).* curvatures;
%         curvatures = abs(curvatures);
%         max_curv = max(curvatures')';
%         min_curv = min(curvatures')'; 
%         curvatures = [max_curv, min_curv];
%         curvatures_gt = curvatures_gt * (1/scale_list(shape_idx));
%         curvatures = curvatures * (1/scale_list(shape_idx));
        gaussian_curvature_gt = curvatures_gt(:,1) .* curvatures_gt(:, 2);
        mean_curvature_gt = (curvatures_gt(:,1) + curvatures_gt(:, 2))/2;
        gaussian_curvature = curvatures(:,1) .* curvatures(:, 2);
        mean_curvature = (curvatures(:,1) + curvatures(:, 2))/2;
        
        diverging_map = [linspace(0, 1, 128)', linspace(0, 1, 128)', ones(128, 1);
                          ones(128, 1), linspace(1, 0, 128)', linspace(1, 0, 128)'];
        colormap(diverging_map);
        caxis([-20, 20])
        if strcmp(curvature_type, 'mean')
            colormap jet
            pc_h.CData = mean_curvature;
        elseif strcmp(curvature_type, 'gaussian')
            pc_h.CData = gaussian_curvature;
        else
%             scale = norm(max(points) - min(points));
%             curvature_range = 2.5;
            curvature_range_min = [-mean(abs(curvatures(:, 2))) - std(abs(curvatures(:, 2))), mean(abs(curvatures(:, 2))) + std(abs(curvatures(:, 2)))];
            curvature_range_max = [-mean(abs(curvatures(:, 1))) - std(abs(curvatures(:, 1))), mean(abs(curvatures(:, 1))) + std(abs(curvatures(:, 1)))];
            curvature_color = curvature_color_mapping(curvatures, curvature_range_max,  curvature_range_min);
            pc_h.CData = curvature_color;  
        end

        image_filename = [curvatures_img_output_path, shape{1}, '.png'];    
        print(image_filename, '-dpng')
    end
    
    if export_curvature_error_vis
        expansion_coeff = 0.1;
        % mapped_gt_c = map_curvatures(curvatures_gt);
        mapped_gt_c = curvatures_gt;
        if strcmp(curv_err_type, 'tanh')
            curv_err = sqrt(sum((tanh(expansion_coeff*mapped_gt_c) - tanh(expansion_coeff*curvatures)).^2, 2)); 
            caxis([0, 2]);
        elseif strcmp(curv_err_type, 'L')
            curv_err = sqrt(sum(((mapped_gt_c - curvatures)./[max(abs(mapped_gt_c(:, 1)), ones(size(mapped_gt_c(:, 1)))), max(abs(mapped_gt_c(:, 2)), ones(size(mapped_gt_c(:, 2))))]).^2, 2)); 
            caxis([0, 5]);
        else
           curv_err = sqrt(sum((mapped_gt_c - curvatures).^2, 2)); 
           caxis([0, 30]);
        end
        rms_k1 = sqrt(nanmean((mapped_gt_c(:, 1) - curvatures(:, 1)).^2));
        rms_k2 = sqrt(nanmean((mapped_gt_c(:, 2) - curvatures(:, 2)).^2));
        
        rms_L_k1 = nanmean((abs(mapped_gt_c(:, 1) - curvatures(:, 1))./max(abs(mapped_gt_c(:, 1)), ones(size(mapped_gt_c(:, 1))))));
        rms_L_k2 = nanmean((abs(mapped_gt_c(:, 2) - curvatures(:, 2))./max(abs(mapped_gt_c(:, 2)), ones(size(mapped_gt_c(:, 2))))));
        
        rms_true_L_k1 = sqrt(nanmean(((mapped_gt_c(:, 1) - curvatures(:, 1))./max(abs(mapped_gt_c(:, 1)), ones(size(mapped_gt_c(:, 1))))).^2));
        rms_true_L_k2 = sqrt(nanmean(((mapped_gt_c(:, 2) - curvatures(:, 2))./max(abs(mapped_gt_c(:, 2)), ones(size(mapped_gt_c(:, 2))))).^2));
        
        rms_tanh_k1 = sqrt(nanmean((tanh(expansion_coeff*mapped_gt_c(:, 1)) - tanh(expansion_coeff*curvatures(:, 1))).^2));
        rms_tanh_k2 = sqrt(nanmean((tanh(expansion_coeff*mapped_gt_c(:, 2)) - tanh(expansion_coeff*curvatures(:, 2))).^2));
        
        colormap jet
         pc_h.CData = curv_err;
%          colormap(jet(256));
         

        active_axes = gca;
        active_axes.Position =  [0, 0.1, 1, 0.9];
        ax_rms_text = axes('position',[0, 0, 1, 0.1]);
        axis off
%         text('string',['k1 RMS = ', num2str(rms_k1, '%.2f')],'units','points', 'position',[30, 48], 'FontSize', 12, 'HorizontalAlignment', 'left')
%         text('string',['k2 RMS = ', num2str(rms_k2, '%.2f')], 'units','points', 'position',[250, 48], 'FontSize', 12, 'HorizontalAlignment', 'left')
        text('string',['k1 L RMS = ', num2str(rms_L_k1, '%.2f')], 'units','points', 'position',[30, 32], 'FontSize', 12, 'HorizontalAlignment', 'left')
        text('string',['k2 L RMS = ', num2str(rms_L_k2, '%.2f')], 'units','points', 'position',[250, 32], 'FontSize', 12, 'HorizontalAlignment', 'left')
%         text('string',['k1 tanh RMS = ', num2str(rms_tanh_k1, '%.2f')], 'units','points', 'position',[30, 16], 'FontSize', 12, 'HorizontalAlignment', 'left')
%         text('string',['k2 tanh RMS = ', num2str(rms_tanh_k2, '%.2f')], 'units','points', 'position',[250, 16], 'FontSize', 12, 'HorizontalAlignment', 'left')
        text('string',['true k1 L RMS = ', num2str(rms_true_L_k1, '%.2f')], 'units','points', 'position',[30, 16], 'FontSize', 12, 'HorizontalAlignment', 'left')
        text('string',['true k2 L RMS = ', num2str(rms_true_L_k2, '%.2f')], 'units','points', 'position',[250, 16], 'FontSize', 12, 'HorizontalAlignment', 'left')   
        image_filename = [curvatures_error_img_output_path, shape{1}, '.png'];    
        print(image_filename, '-dpng')
    end

    close(fig_h);
end
disp('All done!');

end
