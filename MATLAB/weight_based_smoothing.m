function export_visualizations()
clear all
close all
clc

data_path = '/home/sitzikbs/Datasets/pcpnet/';
file_sets_path = '/home/sitzikbs/Datasets/pcpnet/'; % directory containing lists of files ,camera parameters and shape list

results_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/baselines/DeepFit/results/';
% results_path = '/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/excluded_baselines/jet/112_neighbors/';
file_list_file_name = 'testset_high_noise.txt';
shape_list_file_name = 'testset_no_noise.txt';

output_path = [results_path, 'images/noise_removal/'] ;
    if ~exist(output_path,'dir')
        mkdir(output_path)
    end
    
export_type = 'all'; %shape name for single point cloud export or all for all point clouds
export_vid = true;
k_neighbors = 256;
use_subset = false;
point_size = 40;

cam_params = camera_parameters();
xyz_file_list = dir([data_path,'*.xyz']);


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

for shape = shapes_to_export
    disp(['saving ', shape{1}, '...']);
    xyz_file_name = [data_path, shape{1}, '.xyz'];
    normals_gt_file_name = [data_path, shape{1}, '.normals'];
    normals_file_name =  [results_path, shape{1}, '.normals'];
    weights_file_name = [results_path, shape{1}, '.weights'];
    idx_file_name =  [data_path, shape{1}, '.pidx'];

    points = dlmread(xyz_file_name);
    weights = dlmread(weights_file_name);
    normals_gt = dlmread(normals_gt_file_name);
    normals = dlmread(normals_file_name);
    
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

 
    
    fig_h1 = figure('position', [100, 100, 600, 600]);
    ax_h1 = axes('position',[0, 0, 1, 1]);
    set_vis_props(fig_h1, ax_h1);

    if contains(shape, cam_params.name)
        indx=find(cellfun(@contains,repmat({shape},size(cam_params.name)),cam_params.name)==1);
        Reshape(ax_h1, cam_params.r{indx}, cam_params.phi{indx}, cam_params.theta{indx}, [0, 0, 0], cam_params.view_angle{indx}, cam_params.permute{indx});
        start_ang = cam_params.theta{indx};
    else
        Reshape(ax_h1, 10, 45, 35.26, [0, 0, 0], 15, [1, 2, 3]);
        start_ang = 35.26;
    end
        
    xlim([-1, 1]);
    ylim([-1, 1]);
    zlim([-1, 1]);

    
    pc_h = scatter3(points(:, 1), points(:, 2), points(:, 3), point_size, '.');
    mapped_normal_colors = Sphere2RGBCube(-sign(sum(normals_gt.*normals,2)).*normals);
    pc_h.CData = mapped_normal_colors;
%     pc_h.CData = -points(:, 2);
    image_filename = [output_path, shape{1}, 'noisy_n.png'];
    print(image_filename, '-dpng')

    if export_vid
        i = 0;
        % create the video writer with 1 fps
        vid_file_name = [output_path, shape{1}, 'noisy_vid.avi'];
        writerObj = VideoWriter(vid_file_name);
        writerObj.FrameRate = 60;
        % open the video writer
        open(writerObj);
        for ang =[start_ang:0.3:start_ang + 360]
            i = i + 1;
               if contains(shape, cam_params.name)
                    indx=find(cellfun(@contains,repmat({shape},size(cam_params.name)),cam_params.name)==1);
                    Reshape(ax_h1, cam_params.r{indx}, cam_params.phi{indx}, ang, [0, 0, 0], cam_params.view_angle{indx}, cam_params.permute{indx});
                else
                    Reshape(ax_h1, 10, 45, ang, [0, 0, 0], 15, [1, 2, 3]);
                end
            frame = getframe(gcf) ;
            writeVideo(writerObj, frame);
            pause(0.01);
        end
        % close the writer object
        close(writerObj);
    end
    close(fig_h1);  
    
    %visualizer and save smoothed version
    n_points = size(points,1);
    weight_accum = zeros(size(points,1),1);

    ptCloud = pointCloud(points);
    for i=1:n_points
        [nn_indices, nn_dist] = findNearestNeighbors(ptCloud, points(i, :), k_neighbors, 'Sort', true);
        weight_accum(nn_indices) = weight_accum(nn_indices) + weights(i);
    end
    thresh = mean(weight_accum) - 1*std(weight_accum);
    clean_cloud = points(weight_accum > thresh, :);
    mapped_normal_colors = mapped_normal_colors(weight_accum > thresh, :);
    fig_h2 = figure('position', [100, 100, 600, 600]);
    ax_h2 = axes('position',[0, 0, 1, 1]);
    set_vis_props(fig_h2, ax_h2);
    
    if contains(shape, cam_params.name)
        indx=find(cellfun(@contains,repmat({shape},size(cam_params.name)),cam_params.name)==1);
        Reshape(ax_h2, cam_params.r{indx}, cam_params.phi{indx}, cam_params.theta{indx}, [0, 0, 0], cam_params.view_angle{indx}, cam_params.permute{indx});
    else
        Reshape(ax_h2, 10, 45, 35.26, [0, 0, 0], 15, [1, 2, 3]);
    end
    
    pc_clean_h = scatter3(clean_cloud(:, 1), clean_cloud(:, 2), clean_cloud(:, 3), point_size, '.');
    pc_clean_h.CData = mapped_normal_colors;
%     pc_clean_h.CData = -clean_cloud(:, 2);
    image_filename = [output_path, shape{1}, 'smooth_n.png'];
    print(image_filename, '-dpng')
        if export_vid
        i = 0;
        % create the video writer with 1 fps
        vid_file_name = [output_path, shape{1}, 'smooth_vid.avi'];
        writerObj = VideoWriter(vid_file_name);
        writerObj.FrameRate = 60;
        % open the video writer
        open(writerObj);
        for ang =[start_ang:0.3:start_ang + 360]
            i = i + 1;
               if contains(shape, cam_params.name)
                    indx=find(cellfun(@contains,repmat({shape},size(cam_params.name)),cam_params.name)==1);
                    Reshape(ax_h2, cam_params.r{indx}, cam_params.phi{indx}, ang, [0, 0, 0], cam_params.view_angle{indx}, cam_params.permute{indx});
                else
                    Reshape(ax_h2, 10, 45, ang, [0, 0, 0], 15, [1, 2, 3]);
                end
            frame = getframe(gcf) ;
            writeVideo(writerObj, frame);
            pause(0.01);
        end
        % close the writer object
        close(writerObj);
    end
    
    close(fig_h2);
end
disp('All done!');

end
