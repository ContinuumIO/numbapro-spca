
% k is the size of the subgraph
% V is an n by d matrix (vertex by rank), that is computed after the SVD.
% if [Q, D] = eigs(A, d, 'la'), then V = Q*sqrt(D)
% A is the adjacency matrix of the graph(if A is huge, set A=0 as input and uncomment the lines mentioned below)
% eps, delta are approximation error bounds
function [metric_opt supp_opt] = spannogram_DkS_eps_psd(k, V, A, eps, delta)

[n, d] = size(V);
supp_opt = 0;
metric_opt = 0;

Mopt = (1+4/eps)^d;
M = (log(delta)-log(Mopt))./log(1-1./Mopt); % for an error of (1-eps)^2 with probability delta

for i = 1:round(M)
    c = randn(d,1);
    [~, indx] = sort(V*c, 'descend');% descending order
    topk = indx(1:k);
    metric = sum(sum(A(topk,topk)));
    % if A is huge use the following line instead of the above
    % metric = norm(V(topk,:))^2
    if metric>metric_opt
        metric_opt = metric;
        supp_opt = topk;
    end
end

end