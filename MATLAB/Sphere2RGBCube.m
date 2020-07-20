function [ RGB ] = Sphere2RGBCube( V )
%Sphere2RGBCube converts the normalized vector V (representing a point on
%the unit spher) into its corresponding RGB cube values. zero vectors are
%outpt as NaN
%Author:Itzik Ben Sabat sitzikbs[at]gmail.com
%Date: 27.1.2016

if size(V,2) > 3
    V = V';
    transposeglag = true;
else
    transposeglag = false;
end
RGB = zeros(size(V));
V = V./repmat((sqrt(sum(V.^2,2))),1,3); %make sure V is normalized


%Map unit sphere to unit cubev
x = V(:,1);
y = V(:,2);
z = V(:,3);

absx = abs(x);
absy = abs(y);
absz = abs(z);

%Map left and right cube planes  : y=+-1
leftright = and(absy>=absx, absy>=absz);
RGB(leftright,1) = (1 ./ absy(leftright)).*x(leftright);
RGB(leftright,3) = (1 ./ absy(leftright)).*z(leftright);
RGB(and(leftright , y > 0),2) = 1;
RGB(and(leftright , y < 0),2) = -1;

%Map front and back cube planes  : x=+-1
frontback = and(absx >= absy , absx >= absz);
RGB(frontback,2) = (1 ./ absx(frontback)).*y(frontback);
RGB(frontback,3) = (1 ./ absx(frontback)).*z(frontback);
RGB(and(frontback , x > 0),1) = 1;
RGB(and(frontback , x < 0),1) = -1;

%Map top and bottom cube planes  : z=+-1
topbottom = and(absz >= absx , absz >= absy);
RGB(topbottom,1) = (1 ./ absz(topbottom)).*x(topbottom);
RGB(topbottom,2) = (1 ./ absz(topbottom)).*y(topbottom);
RGB(and(topbottom , z > 0),3) = 1;
RGB(and(topbottom , z < 0),3) = -1;

%Map from unit cube to RGB Cube
RGB = 0.5*RGB+0.5;
RGB(all(isnan(V),2),:)=nan; % zero vectors are mapped to black
if transposeglag
    RGB = RGB';
end
end
