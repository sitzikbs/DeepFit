function [xyz_img_output_path, normals_img_output_path, normal_error_img_output_path, curvatures_img_output_path, curvatures_error_img_output_path] = output_directory_setup(output_path, export_point_vis, export_normal_vis, export_normal_error_vis, export_curvature_vis, export_curvature_error_vis)

[xyz_img_output_path, normals_img_output_path,...
    normal_error_img_output_path, curvatures_img_output_path,...
    curvatures_error_img_output_path] = deal({['', '', '', '', '', '']});

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
    if ~exist(normals_img_output_path,'dir')
        mkdir(normals_img_output_path)
    end
end 

% directory for normal errors vis
if export_normal_error_vis
   normal_error_img_output_path = [output_path, 'normal_errors/'];
    if ~exist(normal_error_img_output_path, 'dir')
        mkdir(normal_error_img_output_path)
    end
        fig_h = figure('color','w');
        colormap('parula');
        max_normal_error_ang = 60;
        colorbar('location', 'south', 'Ticks',[0, 1], 'TickLabels',[0, max_normal_error_ang]);
        axis off
        image_filename = [normal_error_img_output_path, 'normal_error_color_bar.png'];    
        print(image_filename, '-dpng')
        close(fig_h);
end 

if export_curvature_vis
    curvatures_img_output_path = [output_path, 'curvatures/'];
    if ~exist(curvatures_img_output_path,'dir')
        mkdir(curvatures_img_output_path)
    end
end 

if export_curvature_error_vis
    curvatures_error_img_output_path = [output_path, 'curvatures_errors/'];
    if ~exist(curvatures_error_img_output_path,'dir')
        mkdir(curvatures_error_img_output_path)
    end
    
    fig_h = figure('color','w');
    colormap('jet');
    max_curv_error = 20; 
    colorbar('location', 'south', 'Ticks',[0, 1], 'TickLabels',[0, max_curv_error]);
    axis off
    image_filename = [curvatures_error_img_output_path, 'curvature_error_color_bar.png'];    
    print(image_filename, '-dpng')
    close(fig_h);
end 

end