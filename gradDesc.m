function [bestW, minRisk, Jw, timeByIter, Ws] = gradDesc(X, Y, lambda, ...
    stepSize, numPointsForStochastic, timeLimitSecs, tolerance)

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

w = randn([size(X, 1) 1]) .* .1; % w is a column vector
iters = 0; time = 0; grd = inf(size(w));
while norm(grd) > tolerance && time <= timeLimitSecs
    start = tic;
    iters = iters + 1;
    grd = findGradient(X, Y, w, lambda, numPointsForStochastic);
    w = w - (stepSize * grd);
%     w = w + (stepSize * grd);
    
    Ws(:, iters) = w; % each w is a column
    Jw(iters) = calculateRisk(X, Y, w, lambda);
    
    fprintf('Jw = %d, norm(grd) = %d\n', Jw(iters), norm(grd));
    
    finish = toc(start);
    time = time + finish;
    timeByIter(iters) = time;
end

if numPointsForStochastic < size(X, 1)
    % If doing SGD, find the w that gave the best risk
    [minRisk, minRiskIdx] = min(Jw);
    bestW = Ws(:, minRiskIdx);
else
    bestW = w;
    minRisk = Jw(iters);
end
