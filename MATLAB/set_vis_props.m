function set_vis_props(fig, ax)
fig.Color = 'w';
fig.Renderer = 'opengl';
% view(3)
% ax.CameraPosition = 3*[-0.5, -1, -1];
% ax.CameraTarget = [0, 0, 0];
% view(-35, -45);
% view(-185, -65);
axis off
daspect([1,1,1]);
% xlim([-1, 1]);
% ylim([-1, 1]);
% zlim([-1, 1]);
% ax.CameraViewAngle=5;
ax.TightInset;
hold all

end

