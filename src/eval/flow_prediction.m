function [flow12] = flow_prediction(pc1, pc2, knn, lambda)
% Predict the flow from pc1 to pc2
deformed_pc1 = pc1;
KNN = knn_search(pc1, knn);
for iter = 1 : 30
    % Compute closest point pairs
    % correspondences = closest_points(deformed_pc1, pc2);
    correspondences = [1:size(deformed_pc1, 2); 1:size(deformed_pc1, 2)];
    for inner = 1 : 4
        rotations = knn_rotation_fitting(pc1, deformed_pc1, KNN);
        deformed_pc1 = pointcloud_fitting(pc1, pc2, correspondences, rotations, KNN, lambda);
    end
    fprintf('.');
    if mod(iter, 10) == 0
        fprintf('\n');
    end
end
flow12 = deformed_pc1 - pc1;
%
function [deformed_pc1] = pointcloud_fitting(pc1, pc2, correspondences, rotations, KNN, lambda)
%
[numP,k] = size(KNN);
sIds = kron(1:numP, ones(1,k));
tIds = reshape(KNN', [1, k*numP]);
%
dpc = pc1(:,sIds) - pc1(:,tIds);
rot = rotations(:,:, sIds);
rot = reshape(rot, [9, size(rot,3)]);
dpc_rot(1,:) = rot(1,:).*dpc(1,:)+rot(4,:).*dpc(2,:)+rot(7,:).*dpc(3,:);
dpc_rot(2,:) = rot(2,:).*dpc(1,:)+rot(5,:).*dpc(2,:)+rot(8,:).*dpc(3,:);
dpc_rot(3,:) = rot(3,:).*dpc(1,:)+rot(6,:).*dpc(2,:)+rot(9,:).*dpc(3,:);
%
J = sparse(ones(2,1)*(1:length(sIds)), [sIds;tIds], [1,-1]'*ones(1,length(sIds)), length(sIds), numP);
L = J'*J;
g = J'*dpc_rot';
%
numC = size(correspondences,2);
J2 = sparse(1:numC, correspondences(1,:), ones(1,numC), numC, numP);
g2 = pc2(:, correspondences(2,:))';
L = L + lambda*(J2'*J2);
g = g + lambda*(J2'*g2);
deformed_pc1 = (L\g)';
e = J*deformed_pc1'-dpc_rot';
e2 = J2*deformed_pc1'-g2;
%fprintf('energy = %f\n', sum(sum(e.*e))+lambda*sum(sum(e2.*e2)));
%

function [rotations] = knn_rotation_fitting(sourcePC, deformedSource, KNN)
%
[numP, k] = size(KNN);
rotations = zeros(3,3,numP);
for vId = 1 : numP
    nIds = KNN(vId, :);
    sourcePatch = sourcePC(:, nIds) - sourcePC(:, vId)*ones(1,k);
    targetPatch = deformedSource(:, nIds) - deformedSource(:, vId)*ones(1,k);
    rotations(:,:,vId) = horn87(sourcePatch, targetPatch);
end
%
function [correspondences] = closest_points(deformed_pc1, pc2)
%
numP_s = size(deformed_pc1, 2);
numP_t = size(pc2, 2);
tIds = knnsearch(pc2', deformed_pc1')';
sIds = knnsearch(deformed_pc1', pc2')';
correspondences = [[1:numP_s;tIds], [sIds;1:numP_t]];

function [KNN] = knn_search(sourcePC, knn)
% Compute the nearest neighbors
[mIdx, mD] = knnsearch(sourcePC', sourcePC', 'K', knn+1);
KNN = mIdx(:,2:(knn+1));

function [R] = horn87(pointsS, pointsT)
%
M = pointsS*pointsT';
[u,sigma,v] = svd(M);
R = v*u';
if det(R) < 0
    u(:,3) = -u(:,3);
    R = v*u';
end
h = 10;
