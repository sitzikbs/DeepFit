function expert_legend(n_experts, colors, spread)
% h = zeros(n_experts,1);
daspect([1,1,1]);
w = 2;
h = 1;
x0 = 0;
y0 = 0;
if strcmp(spread, 'vertical')
     for i=1:n_experts
         rectangle('position', [x0, y0 + h*(i-1), w, h-0.1], 'FaceColor',colors(i,:))
         text('string',num2str(i), 'position',[x0 + w/2, y0 + h*(i-0.5), 0.01], 'FontSize', 14, 'color', 'w')
     end
else
     for i=1:n_experts
         rectangle('position', [x0 + w*(i-1), y0, w - 0.1, h], 'FaceColor',colors(i,:))
         text('string',num2str(i), 'position',[x0 + w*(i-0.5), y0 + h/2, 0.01], 'FontSize', 14, 'color', 'w', 'HorizontalAlignment','center')
     end
end


end

