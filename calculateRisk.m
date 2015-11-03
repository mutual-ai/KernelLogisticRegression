function riskJ = calculateRisk(X, Y, w, lambda)

N = size(X, 1);

sm = 0;
for i = 1:N
    y_i = Y(i);
    k_i = makek_i(i, X);
    sm = sm + logOfSigmoid(y_i * (w' * k_i));
end

riskJ = -1/N * sm + lambda * (w' * w);
