clear all
close all
clc

data_path = '/home/itzik/PycharmProjects/NestiNet/data/pcpnet/';
results_path = '/home/itzik/PycharmProjects/NestiNet/log/experts/pcpnet_results/';
output_path = [results_path, '/images/expert_statistics/'];
output_path_avg_err = [output_path, 'Average expert error/'];
output_path_expert_cnt = [output_path, 'Expert point clount/'];

use_subset = true;
n_experts = 7;
hist_bins = 1:n_experts;
if ~exist(output_path, 'dir')
    mkdir(output_path)
    if ~exist(output_path_avg_err, 'dir')
        mkdir(output_path_avg_err);
    end
    if ~exist(output_path_expert_cnt, 'dir')
        mkdir(output_path_expert_cnt);
    end
end

xyz_file_list = dir([data_path,'*.xyz']);
file_list_to_export = [data_path, 'testset_all.txt'];
shapes_to_export = strsplit(fileread(file_list_to_export));
shapes_to_export = shapes_to_export(~cellfun('isempty',shapes_to_export));  % remove empty cells

accum_expert_count = zeros(1, n_experts);
accume_expert_error  = zeros(1,n_experts);
for shape = shapes_to_export
    
    disp(['processing ', shape{1}, '...']);
    xyz_file_name = [data_path, shape{1}, '.xyz'];
    normals_gt_file_name = [data_path, shape{1}, '.normals'];
    normals_file_name =  [results_path, shape{1}, '.normals'];
    expert_file_name = [results_path, shape{1}, '.experts'];
    idx_file_name =  [data_path, shape{1}, '.pidx'];
    points = dlmread(xyz_file_name);
    points = points - mean(points);
    points = points.* (1./max(sqrt(sum(points.^2, 2))));
    normals_gt = dlmread(normals_gt_file_name);
    normals = dlmread(normals_file_name);
    expert = dlmread(expert_file_name) + 1;
    n_normals = size(normals, 1);
    npoints = size(points,1);
    
    if npoints ~= n_normals
        idxs = dlmread(idx_file_name) + 1;
        points = points(idxs, :);
        normals_gt = normals_gt(idxs, :);
    elseif use_subset
        idxs = dlmread(idx_file_name) + 1;
        points = points(idxs, :);
        normals_gt = normals_gt(idxs, :);
        normals = normals(idxs, :);
        expert = expert(idxs, :);
    end
    
    error = acosd(abs(sum(normals.*normals_gt,2))./ (sqrt(sum(normals.^2,2)).* sqrt(sum(normals_gt.^2,2))));
    shape_error_per_expert = zeros(1, n_experts);
    for i =1:n_experts
        shape_error_per_expert(i) = sum(error(expert == i));
    end
    accume_expert_error = accume_expert_error + shape_error_per_expert;
    edges = [1: n_experts + 1] - 0.5;
    [expert_count, edges] = histcounts(expert, edges);
    accum_expert_count = accum_expert_count + expert_count;
    
    fig_h = figure('color','w', 'numbertitle','off','name','Average expert error');
    ax = axes('fontsize',20, 'fontname','serif', 'xtick',1:n_experts, 'xlim',[1 - 0.5, n_experts + 0.5]);
    xlabel('Expert')
    ylabel('Average error [deg]');
    title('Average expert error')
    hold all
    bar(1:n_experts, shape_error_per_expert./ expert_count)
    image_filename = [output_path_avg_err, shape{1}, '.png'];
    print(image_filename, '-dpng')
    close(fig_h);
    
    fig_h = figure('color','w', 'numbertitle','off','name','Expert point count');
    ax = axes('fontsize',20, 'fontname','serif', 'xtick',1:n_experts, 'xlim',[1 - 0.5, n_experts + 0.5]);
    xlabel('Expert')
    ylabel('Points per expert');
    title('Expert point clount')
    hold all
    bar(1:n_experts, expert_count)
    image_filename = [output_path_expert_cnt, shape{1}, '.png'];
    print(image_filename, '-dpng')
    close(fig_h);
end
avg_error = accume_expert_error./accum_expert_count;
avg_error(isnan(avg_error)) = 0;

% **************************** Visualization *********
figure('color','w', 'numbertitle','off','name','Average expert error');
ax = axes('fontsize',20, 'fontname','serif', 'xtick',1:n_experts, 'xlim',[1 - 0.5, n_experts + 0.5]);
xlabel('Expert');
ylabel('Average error [deg]');
title('Average expert error');
hold all
bar(1:n_experts, avg_error);
image_filename = [output_path, 'Average expert error', '.png'];
print(image_filename, '-dpng');
figure('color','w', 'numbertitle','off','name','Expert point count');
ax = axes('fontsize',20, 'fontname','serif', 'xtick',1:n_experts, 'xlim',[1 - 0.5, n_experts + 0.5]);
xlabel('Expert')
ylabel('Points per expert');
title('Expert point count');
hold all
bar(1:n_experts, accum_expert_count);
image_filename = [output_path, 'Expert point count', '.png'];
print(image_filename, '-dpng');
