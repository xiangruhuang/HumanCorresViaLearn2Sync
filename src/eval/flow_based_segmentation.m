function [segments] = flow_based_segmentation(sourcePC, flowPC, knn, alpha)
% Compute the segmentation based on the flow 'flowPC' applied to sourcePC
% Compute the k nearest neighbor graph
neighborIds = knn_search(sourcePC, knn);
% Compute the residual of rigid alignments in the local neighborhoods
residuals = knn_rigidity_fitting(sourcePC, sourcePC+flowPC, neighborIds);
% Segmentation based on region-growing
patches = patch_decomposition(neighborIds, residuals);
%
segments = region_growing(sourcePC, sourcePC+flowPC, patches, neighborIds, alpha);

function [KNN] = knn_search(sourcePC, knn)
% Compute the nearest neighbors
[mIdx, mD] = knnsearch(sourcePC', sourcePC', 'K', knn+1);
KNN = mIdx(:,2:(knn+1));
%
function [residuals] = knn_rigidity_fitting(sourcePC, targetPC, KNN)
%
numP = size(sourcePC, 2);
residuals = zeros(numP, 1);
for pId = 1 : numP
    nIds = KNN(pId,:);
    sourcePatch = sourcePC(:,nIds) - sourcePC(:, pId)*ones(1, length(nIds));
    targetPatch = targetPC(:,nIds) - targetPC(:, pId)*ones(1, length(nIds));
    residuals(pId) = horn87(sourcePatch, targetPatch);
end
%
function [patches] = patch_decomposition(KNN, residuals)
%
numV = size(KNN, 1);
parent = -ones(1, numV);
%
for vId = 1 : numV
    nIds = KNN(vId, :);
    min_res = 1e10;
    min_nId = -1;
    for j = 1 : length(nIds)
        nId = nIds(j);
        if residuals(nId) < min_res
            min_res = residuals(nId);
            min_nId = nId;
        end
    end
    if residuals(vId) > min_res || (residuals(vId)== min_res && vId > min_nId)
        parent(vId) = min_nId;
    end
end
for vId = 1:numV
    if parent(vId) == -1
        continue;
    end
    iter = vId;
    while 1
        if parent(iter) == -1
            parent(vId) = iter;
            break;
        end
        iter = parent(iter);
    end
end
cId = 0;
for vId = 1 : numV
    if parent(vId) == -1
        cId = cId + 1;
        patches{cId} = [vId, find(parent == vId)];
    end
end
%
function [segments] = region_growing(sourcePC, targetPC, patches, neighborIds, alpha)
%
numV = size(sourcePC, 2);
graph = sparse(numV, numV);
for vId = 1 : numV
    graph(vId, neighborIds(vId,:)) = 1;
end
graph = max(graph, graph');
%
numP = length(patches);
patchAdjGraph = zeros(numP, numP);
for p1Id = 1 : numP
    for p2Id = (p1Id+1) : numP
        patchAdjGraph(p1Id, p2Id) = sum(sum(graph(patches{p1Id}, patches{p2Id})));
    end
end
[s,t,v] = find(patchAdjGraph);
for off = 1 : length(v)
    sId = s(off);
    tId = t(off);
    ids = [patches{sId}, patches{tId}];
    v(off) = mean_fitting_error(sourcePC(:,ids), targetPC(:,ids));
end
threshold = median(v)*alpha;
currentGraph = sparse(s,t,v,numP, numP);
while 1
    [s,t,v] = find(currentGraph);
    % find minEdge
    [minScore, off] = min(v);
    if minScore > threshold
        break;
    end
    nextPatch1 = s(off);
    nextPatch2 = t(off);
    numP = size(currentGraph, 1);
    order = [1:(nextPatch1-1),(nextPatch1+1):(nextPatch2-1), (nextPatch2+1):numP];
    mergeIds = [nextPatch1, nextPatch2];
    v = sum(currentGraph(order, mergeIds)')+sum(currentGraph(mergeIds,order));
    currentGraph = currentGraph(order, order);
    currentGraph = [currentGraph, v'];
    currentGraph(numP-1,:) = sparse(1,numP-1);
    tp = [patches{nextPatch1}, patches{nextPatch2}];
    patches = patches(order);
    patches{numP-1} = tp;
    [recomputeIds, notused, notused2] = find(currentGraph(:, numP-1));
    for i = 1 : length(recomputeIds)
        firstPId = recomputeIds(i);
        ids = [patches{firstPId}, patches{numP-1}];
        v = mean_fitting_error(sourcePC(:,ids), targetPC(:,ids));
        currentGraph(firstPId, numP-1) = v;
    end
end
segments = patches;

function [error] = mean_fitting_error(sourcePatch, targetPatch)
%
numP = size(sourcePatch, 2);
sourcePatch = sourcePatch - mean(sourcePatch')'*ones(1,numP);
targetPatch = targetPatch - mean(targetPatch')'*ones(1,numP);
error = horn87(sourcePatch, targetPatch);