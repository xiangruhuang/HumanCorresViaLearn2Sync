function [] = draw_segments(sourcePC, segments)
%
hold on;
daspect([1,1,1]);
for sId = 1 : length(segments)
    seg = segments{sId};
    seg = sourcePC(:, seg);
    plot3(seg(1,:), seg(2,:), seg(3,:), 's', 'markerFaceColor', rand(1,3));
end