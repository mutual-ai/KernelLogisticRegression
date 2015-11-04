function k_i = makek_i(i, dataset, clearPersistent)
% MUST specify clearPersistent = true if dataset is different!

persistent K
if isempty(K) || (nargin > 2 && clearPersistent)
    K = nan(size(dataset, 1), size(dataset, 1)); 
    [~] = rbfKernel(dataset(1, :), dataset(2, :), dataset, true);
end

x_i = dataset(i, :);
if isnan(K(:, i))
    k_i = nan(size(dataset, 1), 1);
    for j = 1:size(dataset, 1)
        x_j = dataset(j, :);
        k_i(j) = rbfKernel(x_i, x_j, dataset);
    end
    K(:, i) = k_i;
else
    k_i = K(:, i);
end
