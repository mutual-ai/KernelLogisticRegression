function [gradient] = findGradient(X, Y, w, p)
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
for pp = 1:p
    i = stochasticPtsToTake(pp);
    x_i = X(i, :);
    y_i = Y(i);
    for j = 1:length(w)
        % Based on http://deeplearning.stanford.edu/wiki/index.php/
        % Logistic_Regression_Vectorization_Example and 
        % http://cs229.stanford.edu/notes/cs229-notes1.pdf (p. 18), but it 
        % looks like they're both maximizing the likelihood, so signs are
        % flipped. 
        
        k_i = makek_i(i, X);
%         sm(j) = sm(j) + (y_i * (w' * k_i)) * x_i(j);
        sm(j) = sm(j) + (y_i * (w' * k_i)) * k_i(j); % Who the fuck knows
        % Check derivation -- this might be wrong, esp wrt where y_i goes
        % Check http://www.cs.cmu.edu/~awm/15781/assignments/hw3_sol.pdf
        % Also what about lambda/regularization?
    end
end

gradient = (-1/p * sm)';
end
