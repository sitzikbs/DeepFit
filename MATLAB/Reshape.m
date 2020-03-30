function Reshape(ax, r, theta, phi, c0, viewangle, permute)
if nargin < 5 
    c0 = [0, 0, 0];
    viewangle = 20;
    permute=[0, 1, 2];
end

CamX=r*sind(theta)*sind(phi)  + c0(1); CamY=r*cosd(theta) + c0(2); CamZ=r*sind(theta)*cosd(phi) + c0(3);
UpX=-cosd(theta)*sind(phi);UpY=sind(theta);UpZ=-cosd(theta)*cosd(phi);
cam_pos = [CamX,CamY,CamZ];
up_dir = [UpX,UpY,UpZ];
cam_pos = cam_pos(permute);
up_dir = up_dir(permute);
set(ax,'CameraPosition',cam_pos,'CameraTarget',c0,...
    'CameraUpVector',up_dir, 'CameraViewAngle', viewangle);
end