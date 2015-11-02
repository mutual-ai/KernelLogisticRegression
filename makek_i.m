function k_i = makek_i(x_i, dataset)

k_i = nan(size(x_i, 1), 1);
for j = 1:size(dataset, 1)
    x_j = dataset(j, :);
    k_i(j) = rbfKernel(x_i, x_j, dataset);
end
