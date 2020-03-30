function Cdata = curvature_color_mapping(PrincipalCurvatures, k1_range, k2_range)
% map the principal curvatures to color space where pp is red, nn is blue,
% pn is green and 00 is white (p=positiv,m n=negative)
k1min = k1_range(1);
k1max = k1_range(2);
k2min = k2_range(1);
k2max = k2_range(2);
PrincipalCurvatures(find(PrincipalCurvatures(:, 1)>k1max), 1)=k1max;
PrincipalCurvatures(find(PrincipalCurvatures(:, 1)<k1min), 1)=k1min;
PrincipalCurvatures(find(PrincipalCurvatures(:, 2)>k2max), 2)=k2max;
PrincipalCurvatures(find(PrincipalCurvatures(:, 2)<k2min), 2)=k2min;

[X, Y] = meshgrid([k1min 0 k1max], [k2min 0 k2max]);

red_dist=       [0 0.5  0; 0.5 1   1 ; 0   1   1 ];
green_dist =    [0  1   1;   1 1   1 ; 1   1   0 ];
blue_dist =     [1  1   0;   1 1  0.5; 0  0.5  0 ];


Xq=PrincipalCurvatures(:, 1);
Yq=PrincipalCurvatures(:, 2);
Cdata(:,1)=interp2(X,Y,red_dist,Xq,Yq);
Cdata(:,2)=interp2(X,Y,green_dist,Xq,Yq);
Cdata(:,3)=interp2(X,Y,blue_dist,Xq,Yq);
end