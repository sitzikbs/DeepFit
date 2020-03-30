function [points] = ScanNet_depth2xyz(depth_img, intrinsic, pose)
% converts from depth map to point cloud
depth_shift = 1;
camera_to_world = pose;
intrinsicInv  = inv(intrinsic);
depth_height = size(depth_img, 1);
depth_width = size(depth_img, 2);
points = zeros(depth_height* depth_width, 3);
i = 1;
  for y = 1: depth_height 
    for x = 1: depth_width 
      if depth_img(y, x) ~= 0
            d = double(depth_img(y, x)) / depth_shift;
            camera_pos = intrinsicInv * [ x * d; y * d; d; 0.0];
            world_pos = camera_to_world* camera_pos;
            points(i, :) = world_pos(1:3);
            i = i+1;
      end
    end
  end
  points( ~any(points,2), : ) = [];
end

