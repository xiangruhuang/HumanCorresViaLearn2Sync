function [res] = horn87(pointsS, pointsT)
%
M = pointsS*pointsT';
[u,sigma,v] = svd(M);
R = v*u';
if det(R) < 0
    u(:,3) = -u(:,3);
    R = v*u';
end
%
tp = R*pointsS-pointsT;
res = max(sqrt(sum(tp.*tp)));