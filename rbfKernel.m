function output = rbfKernel(x_i, x_j, dataset)

persistent kappa2;
if isempty(kappa2)
    s = 0;
    for i = 1:size(dataset, 1)
        for j = 1:size(dataset, 1)
            s = s + norm(dataset(i, :) - dataset(j, :))^2;
        end
    end
    kappa2 = 1/size(dataset, 1)^2 * s;
end

output = exp(-norm(x_i - x_j)^2 / kappa2);
