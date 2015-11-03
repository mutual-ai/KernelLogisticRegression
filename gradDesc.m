function [bestW, minRisk] = gradDesc(X, Y, lambda, stepSize, ...
    numPointsForStochastic, timeLimitSecs, tolerance)

% Default tolerance: 1e-2 (from problem 5 description)
if nargin < 7
    tolerance = 1e-2;
end
% To specify a time limit, you must specify number of points for stochastic
if nargin < 6
    timeLimitSecs = Inf;
end
% Number of points to use to estimate gradient for stochastic gradient
% descent. If not specified, defaults to size(X, 1) (i.e. non-stochastic
% gradient descent).
if nargin < 5
    numPointsForStochastic = size(X, 1);
end

w = randn([1 size(X, 1)])' .* .1; % Check this?
iters = 0; time = 0; grd = inf(size(w)); start = cputime;
while norm(grd) > tolerance && time <= timeLimitSecs
    iters = iters + 1;
    grd = findGradient(X, Y, w, numPointsForStochastic);
    w = w - (stepSize * grd);
    
    Ws(iters, :) = w;
    Jw(iters) = calculateRisk(X, Y, w, lambda)
    time = (cputime - start)*60; % cputime given in minutes
end

if numPointsForStochastic < size(X, 1)
    % If doing SGD, find the w that gave the best risk
    [minRisk, minRiskIdx] = min(Jw);
    bestW = Ws(minRiskIdx);
else
    bestW = w;
    minRisk = Jw(iters);
end
