function [img] = ScanNet_world2cam_normals(points, prop, depth_img, intrinsic, pose)
%project point properties back to image plane
%world2camer is pose matrix given in the dataset
world2camera = inv(pose);

depth_height = size(depth_img, 1);
depth_width = size(depth_img, 2);
n_points = size(points, 1);

img = zeros(depth_height, depth_width, 3);
for i =1:n_points
    pixel_idx = intrinsic * world2camera * [points(i,:)'; 1];
    pixel_idx = pixel_idx./pixel_idx(3);
    x = round(pixel_idx(1));
    y = round(pixel_idx(2));
    if x > 0 && y > 0 && x <= depth_width && y <= depth_height
        img(y, x, :) = reshape(prop(i, :), [1, 1, 3]);
    end
end
end

