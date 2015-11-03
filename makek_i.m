function k_i = makek_i(i, dataset)

persistent K origDataset;
if isempty(origDataset)
    origDataset = dataset;
end
if isempty(K) || ~isequal(dataset, origDataset)
    K = nan(size(dataset, 1), size(dataset, 1)); 
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
