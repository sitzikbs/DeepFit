function Reshape(ax, r, theta, phi)
CamX=r*sind(theta)*sind(phi); CamY=r*cosd(theta); CamZ=r*sind(theta)*cosd(phi);
UpX=-cosd(theta)*sind(phi);UpY=sind(theta);UpZ=-cosd(theta)*cosd(phi);
set(ax,'CameraPosition',[CamX,CamZ,CamY],'CameraTarget',[0 0 0],...
    'CameraUpVector',[UpX,UpZ,UpY],'CameraViewAngle', 60);
end