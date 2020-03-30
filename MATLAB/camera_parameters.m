function camera_params = camera_parameters()

camera_params.name = {'Liberty','netsuke', 'galera', 'Cup', 'pipe_curve', 'column100k'};
camera_params.r = {10, 10, 10, 10, 10, 10};
camera_params.phi = {45, 90-35.26, 45 + 180, 45, 45, 45};
camera_params.theta = {35.26, 90, 35.26, 35.26, 35.26+90, 35.26};
camera_params.view_angle = {10, 10, 10, 15, 10, 10};
camera_params.x_scale = {1, 1, 1, 1, 1, 1};
camera_params.y_scale = {1, 1, -1, 1, 1, 1};
camera_params.z_scale = {1, 1, 1, 1, 1, 1};
camera_params.permute = {[1,2,3], [1,2,3], [1,2,3], [1, 3, 2], [1, 2, 3], [1, 2, 3]};
end