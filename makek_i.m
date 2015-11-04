function k_i = makek_i(i, dataset, clearPersistent, x_i)
% clearPersistent (optional): true to clear cache (MUST specify true if 
% dataset is different!)
% x_i (optional): data point x if not drawing it from dataset(i, :); 
% ignores i if x_i is supplied and does not cache result

persistent K
if isempty(K) || (nargin > 2 && clearPersistent)
    K = nan(size(dataset, 1), size(dataset, 1)); 
end

if nargin > 3
    i = 0;
    assert(length(x_i) == size(dataset, 2));
else
    x_i = dataset(i, :);
end

if i == 0 || any(isnan(K(:, i)))
    k_i = nan(size(dataset, 1), 1);
    for j = 1:size(dataset, 1)
        x_j = dataset(j, :);
        k_i(j) = rbfKernel(x_i, x_j, dataset);
    end
    
    if i ~= 0
        K(:, i) = k_i;
    end
else
    k_i = K(:, i);
end
