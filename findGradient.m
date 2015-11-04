function [gradient] = findGradient(X, Y, w, lambda, p)
% w: current weights
% p: number of points to use to estimate gradient for stochastic gradient
% descent. If not specified, defaults to size(X, 1) (i.e. non-stochastic
% gradient descent.)

if nargin < 4
    p = size(X, 1);
end

N = size(X, 1); % Number of data points (rows of X)
stochasticPtsToTake = randperm(N, p);
sm = zeros([1 length(w)]);
% sm = 0;
for pp = 1:p
    i = stochasticPtsToTake(pp);
    x_i = X(i, :);
    y_i = Y(i);
    k_i = makek_i(i, X);
    
%     sm = sm + 1 - sigmoid(y_i * w' * k_i) + 2 * lambda * w';
%     sm = sm + 1 - sigmoid(y_i * w' * k_i);
    for j = 1:length(w)
%         sm(j) = sm(j) + 1 - sigmoid(y_i * w' * k_i) + 2 * lambda * w(j);
        sm(j) = sm(j) + 1 - sigmoid(y_i * w(j) * k_i(j));
    end
end

gradient = (-1/p * sm)' + 2 * lambda * w;
end
