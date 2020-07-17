clear all
close all
clc

addpath('/home/sitzikbs/Datasets/NYU_V2/toolbox') % path to nyu v2 toolbox
data_path = '/home/sitzikbs/Datasets/NYU_V2/';
results_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/jetnet_nci_new3/ablations/Deepfit_knn_lr0.001_sigmoid_cr_log_d3_p256_Lsin/results/non_pcpnet/128/';
output_path = [results_path, 'images/'];

export_type = 'all'; % set to 'all' to export all files in the results firectory
% Choose if to export all files in the results folder or just a subset
if strcmp(export_type, 'all')
    file_list_struct =  dir([results_path, '*.normals']);
    file_list=cell(size(file_list_struct, 1), 1);
    for i=1:size(file_list, 1)
        file_list{i} = file_list_struct(i).name(1:end-8);
    end
    shapes_to_export = file_list;
else
    file_list_file_name = 'testset_all.txt';
    file_list_to_export = [data_path, file_list_file_name];
    shapes_to_export = strsplit(fileread(file_list_to_export));
    shapes_to_export = shapes_to_export(~cellfun('isempty',shapes_to_export));  % remove empty cells
end

export_point_vis = true;
export_normal_vis = true;
export_curvature_vis = true;
export_expert_vis = false;
curvature_type = 'both';

n_experts = 7;

use_subset = false;

if use_subset
    point_size = 400;
else
    point_size = 10;
end

if ~exist(output_path, 'dir')
    mkdir(output_path)
end

% directory for exporting XYZ vis
if export_point_vis
    xyz_img_output_path = [output_path, 'xyz/'];
    if ~exist(xyz_img_output_path,'dir')
        mkdir(xyz_img_output_path)
    end
end
% directory for exporting normal vis
if export_normal_vis
    normals_img_output_path = [output_path, 'normals/'];
    normals_img_2D_output_path = [output_path, 'normals/2D/'];
    if ~exist(normals_img_output_path,'dir')
        mkdir(normals_img_output_path);
    end
    if ~exist(normals_img_2D_output_path,'dir')
        mkdir(normals_img_2D_output_path);
    end
end
% directory for exporting experts vis
if export_expert_vis
    expert_img_output_path = [output_path, 'experts/'];
    experts_img_2D_output_path = [output_path, 'experts/2D/'];
    if ~exist(expert_img_output_path, 'dir')
        mkdir(expert_img_output_path)
    end
    if ~exist(experts_img_2D_output_path,'dir')
        mkdir(experts_img_2D_output_path);
    end
end

if export_curvature_vis
    curvatures_img_output_path = [output_path, 'curvatures/'];
    if ~exist(curvatures_img_output_path,'dir')
        mkdir(curvatures_img_output_path)
    end
    curvatures_img_2D_output_path = [output_path, 'curvatures/2D/'];
    if ~exist(curvatures_img_2D_output_path,'dir')
        mkdir(curvatures_img_2D_output_path);
    end
end

xyz_file_list = dir([data_path,'*.xyz']);


% for shape = shapes_to_export
for i = 1:size(shapes_to_export,1)
    shape = shapes_to_export{i};
    disp(['saving ', shape, '...']);
    xyz_file_name = [data_path, shape, '.xyz'];
    pose_file_name =  [data_path, shape, '.pose'];
    normals_file_name =  [results_path, shape, '.normals'];
    expert_file_name = [results_path, shape, '.experts'];
    
    original_points = dlmread(xyz_file_name);
    normals = dlmread(normals_file_name);
    normals = normals./sqrt(sum(normals.^2, 2));
    if use_subset
        idxs = randi([1 npoints],1,50000);
        original_points = original_points(idxs, :);
        normals = normals(idxs, :);
    end
    
    original_cam_points = depth_world2rgb_world(original_points);
    cam_points = original_cam_points - mean(original_cam_points);
    cam_points = cam_points * (1/max(sqrt(sum(cam_points.^2, 2))));
    M =[-1,0,0;0,1,0;0,0,1];
    cam_points =  M * cam_points';
    cam_points = cam_points';
    if exist(pose_file_name, 'file')
        angles = dlmread(pose_file_name);
        phi = angles(1);
        theta = angles(2);
    else
        theta = 90 - 15;
        phi = 0;
    end
    r = 5;
    
    npoints = size(original_cam_points,1); % new number of points
    fig_h = figure('color','w');
    ax_h = axes('position',[0, 0, 1, 1]);
    cam_points = cam_points (1:npoints,:);
    
    %%%%%%%%%%%%% point cloud visualization %%%%%%%%%%%%%%
    pc_h = scatter3(cam_points(:, 1), cam_points(:, 2), cam_points(:, 3), 1, '.','cdata', cam_points(:, 2));
    Reshape(gca, r, theta , phi, [0, 0, 0], 20, [1,2,3]);
    daspect([1,1,1]);
    axis off
    if export_point_vis
        image_filename = [xyz_img_output_path, shape, '.png'];
        print(image_filename, '-dpng')
    end
    
    %%%%%%%%%%%%% normal visualization %%%%%%%%%%%%%%
    if export_normal_vis
        n_normals = size(normals, 1);
        positive_direction = repmat([0, 0, 0], size(original_cam_points,1),1) - original_cam_points; %align
        normals = sign(sum(positive_direction.*normals, 2)).* normals;
        normals(:, 3) = -normals(:, 3);
        Ry_90 = makehgtform('yrotate',-pi/2);
        normals = (Ry_90(1:3, 1:3) * normals')';

        mapped_normal_colors = Sphere2RGBCube(normals);
        
        %         pc_h = scatter3(cam_points(:, 1), cam_points(:, 2), cam_points(:, 3), 1, '.','cdata',mapped_normal_colors);
        set(pc_h, 'cdata', mapped_normal_colors);
        axis off
        Reshape(gca, r, theta , phi, [0, 0, 0], 20, [1,2,3]);
        daspect([1,1,1]);
        image_filename = [normals_img_output_path, shape, '.png'];
        print(image_filename, '-dpng')
        
        % 2D projection
        camera_params
        x = round(original_points(:, 1) * fx_d ./ original_points(:, 3) + cx_d);
        y = round(original_points(:, 2) * fy_d ./ original_points(:, 3) + cy_d);
        normal_img = zeros(480, 640, 3);
        [mask sz] = get_projection_mask();
        for j = 1:length(x(:))
            normal_img( y(j), x(j), 1:3) =  reshape(mapped_normal_colors(j, :), [1, 1, 3]);
        end
        normal_img = normal_img(45:471, 41:601, :); %crop out black frame
        %          imshow(normal_img);
        image_filename = [normals_img_2D_output_path, shape, '.png'];
        imwrite(normal_img, image_filename );
    end
    
    if export_expert_vis
        expert = dlmread(expert_file_name);
        n_colors = n_experts;
        colors = distinguishable_colors(n_colors);
        ax_h = gca;
        fig_h = gcf;
        ax_h.Position = [0, 0, 0.8, 1];
        active_axes = ax_h;
        axes('position',[0.8, 0, 0.2, 1]);
        axis off
        expert_legend(n_experts, colors, 'vertical');
        fig_h.CurrentAxes  = active_axes;
        colormap(colors)
        %         pc_h = scatter3(cam_points(:, 1), cam_points(:, 2), cam_points(:, 3), 1, '.','cdata',colors( expert+1, :));
        set(pc_h, 'cdata', colors( expert+1, :));
        Reshape(gca, r, theta , phi, [0, 0, 0]);
        daspect([1,1,1]);
        axis off
        image_filename = [expert_img_output_path, shape, '.png'];
        print(image_filename, '-dpng')
        
        % export 2D expert image
        camera_params
        x = round(original_points(:, 1) * fx_d ./ original_points(:, 3) + cx_d);
        y = round(original_points(:, 2) * fy_d ./ original_points(:, 3) + cy_d);
        experts_img = zeros(480, 640, 3);
        [mask sz] = get_projection_mask();
        for i = 1:length(x(:))
            experts_img(y(i), x(i), 1:3) =  reshape(colors(expert(i)+1, :), [1, 1, 3]);
        end
        experts_img = experts_img(45:471, 41:601, :); %crop out black frame
        imshow(experts_img);
        image_filename = [experts_img_2D_output_path, shape, '.png'];
        imwrite(experts_img, image_filename );
    end
    
    if export_curvature_vis
        curvs_file_name =  [results_path, shape, '.curv'];
        curvatures = dlmread(curvs_file_name);
        curvatures = curvatures(:, 1:2);
        normals = dlmread(normals_file_name);
        normals = normals./sqrt(sum(normals.^2, 2));
        positive_direction = - original_cam_points; %align
        orientation_sign = sign(sum(positive_direction.*normals, 2));
        normals_gt = -original_cam_points ./ repmat(sqrt(sum(original_cam_points'.^2))', 1, 3); % assuming camera position at 0
        
        curvatures = repmat(orientation_sign, 1, 2).* curvatures;
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
            std_factor = 2;
            curvature_range_min = [-mean(abs(curvatures(:, 2))) - std_factor * std(abs(curvatures(:, 2))), mean(abs(curvatures(:, 2))) + std_factor * std(abs(curvatures(:, 2)))];
            curvature_range_max = [-mean(abs(curvatures(:, 1))) - std_factor * std(abs(curvatures(:, 1))), mean(abs(curvatures(:, 1))) + std_factor * std(abs(curvatures(:, 1)))];
            curvature_color = curvature_color_mapping(curvatures, curvature_range_max,  curvature_range_min);
            pc_h.CData = curvature_color;  
        end

        image_filename = [curvatures_img_output_path, shape, '.png'];    
        print(image_filename, '-dpng')
        
                % 2D projection
        camera_params
        x = round(original_points(:, 1) * fx_d ./ original_points(:, 3) + cx_d);
        y = round(original_points(:, 2) * fy_d ./ original_points(:, 3) + cy_d);
        curvature_img = zeros(480, 640, 3);
        [mask sz] = get_projection_mask();
        for j = 1:length(x(:))
            curvature_img( y(j), x(j), 1:3) =  reshape(curvature_color(j, :), [1, 1, 3]);
        end
        curvature_img = curvature_img(45:471, 41:601, :); %crop out black frame
        %          imshow(normal_img);
        image_filename = [curvatures_img_2D_output_path, shape, '.png'];
        imwrite(curvature_img, image_filename );
        
    end
    close(fig_h);
end
disp('All done!');