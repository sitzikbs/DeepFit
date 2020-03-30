function surf_h=plot_jet(beta, order, points, trans)
rangex = [min(points(:, 1)), max(points(:, 1))];
rangey = [min(points(:, 2)), max(points(:, 2))];
dx = rangex(2)-rangex(1);
dy = rangey(2)-rangey(1);
[x, y] = meshgrid(rangex(1):0.1*dx:rangex(2), rangey(1):0.1*dy:rangey(2));
if order == 2
    z = beta(1)*x + beta(2)*y  + beta(3) * x.^2 + beta(4) * y.^2 + beta(5) * x.*y + beta(6) ;
elseif order == 3 
    z = beta(1)*x + beta(2)*y  + beta(3) * x.^2 + beta(4) * y.^2 + beta(5) * x.*y + beta(6) * x.^3 + beta(7) * y.^3 + beta(8) * x.^2 .* y + beta(9) * x .* y.^2 + beta(10);
end
res = size(x);
tj_points =  [x(:), y(:), z(:)] * squeeze(trans);
% tj_points = tj_points';
hold all
surf_h = surf(reshape(tj_points(:, 1), res), ...
    reshape(tj_points(:, 2), res), reshape(tj_points(:, 3), res) ,'facecolor', 'r','facealpha',0.3, 'edgealpha',0.2);
end
