clear all
close all
clc
frames_path = '/home/itzik/PycharmProjects/NestiNet/data/ScanNet/frames/';
data_path = '/home/sitzikbs/Datasets/NYU_V2/ScanNet/';
results_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/jetnet_nci_new3/ablations/Deepfit_knn_lr0.001_sigmoid_cr_log_d3_p256_Lsin/results/non_pcpnet/';
output_path = [results_path, 'images/'];
file_list_file_name = 'testset_all.txt';

export_point_vis = true;
export_normal_vis = true;
export_expert_vis = true;
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
        mkdir(normals_img_output_path)
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

xyz_file_list = dir([data_path,'*.xyz']);
file_list_to_export = [data_path, file_list_file_name];
shapes_to_export = strsplit(fileread(file_list_to_export));
shapes_to_export = shapes_to_export(~cellfun('isempty',shapes_to_export));  % remove empty cells

for shape = shapes_to_export
    disp(['saving ', shape{1}, '...']);
    xyz_file_name = [data_path, shape{1}, '.xyz'];
    ply_file_name = [data_path, shape{1}, '.ply'];
%     pose_file_name =  [data_path, shape{1}, '.pose'];
    normals_file_name =  [results_path, shape{1}, '.normals'];
    expert_file_name = [results_path, shape{1}, '.experts'];
    
    
    scene_info = strsplit(shape{1},'_');
    scene = ['scene', scene_info{2},'_',scene_info{3}];
    frame_idx = scene_info{5}; % extract frame number
    image_file_name = [frames_path,scene,'/color/',frame_idx, '.jpg'];
    depth_image_file_name = [frames_path, scene,'/depth/', frame_idx,  '.png'];
    pose_file_path = [frames_path, scene,'/pose/', frame_idx,  '.txt'];
    intrinsics_path = [frames_path, scene, '/intrinsic_depth.txt'];
    
    depth_img = imread(depth_image_file_name);
    intrinsic = dlmread(intrinsics_path);
    pos_mat = dlmread(pose_file_path);
    
    %       pc = pcread(ply_file_name);
    %       original_points = pc.Location;
    %       clear pc.Color %remove RGB color
    original_points = dlmread(xyz_file_name);
    
    %      points = dlmread(xyz_file_name);
    points = original_points - mean(original_points);
    points = points.* (1./max(sqrt(sum(points.^2, 2))));
    pc = pointCloud(points);
    
    normals = dlmread(normals_file_name);
    n_normals = size(normals, 1);
    npoints = size(points,1);
    
    camer_to_world = pos_mat(1:3, 1:3);
    world_to_camera = inv(camer_to_world);
    cam_ij = world_to_camera * original_points';
    
    cam_pos = [pos_mat(1, 4); -pos_mat(2, 4); pos_mat(3, 4)];
    cam_target = cam_pos + pos_mat(3, 1:3)';
    cam_up =  pos_mat(2, 1:3)';
    if use_subset
        idxs = randi([1 npoints],1,50000);
        points = points(idxs, :);
        normals = normals(idxs, :);
    end
    
    npoints = size(points,1); % new number of points
    fig_h = figure('color','w', 'units','normalized','outerposition',[0 0 1 1]);
    ax_h = axes('position',[0, 0, 1, 1]);

    points = points (1:npoints,:);
    normals = normals(1:npoints,:);
    Ry_90 = makehgtform('yrotate', pi/2); % wierd rotation between real and synth - should check source
    normals = (Ry_90(1:3, 1:3) * normals')';
    normals(:,3) = -normals(:,3) ;
    positive_direction = repmat([pos_mat(1, 4); pos_mat(2, 4); pos_mat(3, 4)]', size(points,1),1) - points;
    normals = -sign(sum(positive_direction.*normals, 2)).* normals; %orient normals towards viewer

    
    pcshow(pc);
    hold all
    set(ax_h,'CameraPosition',cam_pos);
    axis off
    if export_point_vis
        image_filename = [xyz_img_output_path, shape{1}, '.png'];
        print(image_filename, '-dpng')
    end
    
    if export_normal_vis
        mapped_normal_colors = Sphere2RGBCube(normals);
        pc.Color = im2uint8(mapped_normal_colors);
        
        pcshow(pc);
        set(ax_h,'CameraPosition',cam_pos);
        axis off
        image_filename = [normals_img_output_path, shape{1}, '.png'];
        print(image_filename, '-dpng')
        
        %2D image visualization
        points = ScanNet_depth2xyz(depth_img, intrinsic, pos_mat); %reconstruct the original point cloud since the one in the file is normalized
        normal_img = ScanNet_world2cam_normals(points,mapped_normal_colors, depth_img, intrinsic, pos_mat); %project the normals back to the image plane
        image_filename = [normals_img_2D_output_path, shape{1}, '.png'];
        imwrite(normal_img, image_filename );
    end
    
    
    if export_expert_vis
        expert = dlmread(expert_file_name);
        n_colors = n_experts;
        colors = distinguishable_colors(n_colors);
        pc.Color = im2uint8(colors( expert+1, :));
        pcshow(pc);
        gca.position = [0, 0, 0.8, 1];
        active_axes = gca;
        axes('position',[0.8, 0, 0.2, 1]);
        axis off
        gcf.CurrentAxes  = active_axes;
        colormap(colors);
        
        set(ax_h,'CameraPosition',cam_pos);
        axis off
        image_filename = [expert_img_output_path, shape{1}, '.png'];
        print(image_filename, '-dpng')
        
        %2D visualization
        experts_img = ScanNet_world2cam_normals(points,colors(expert+1, :), depth_img, intrinsic, pos_mat); %project the normals back to the image plane
        image_filename = [experts_img_2D_output_path, shape{1}, '.png'];
        imwrite(experts_img, image_filename );
    end
    close(fig_h);
end
disp('All done!');