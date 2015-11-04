%% Update path -- replace with location of dataset, if necessary
addpath('../Provided files/');
load('data1.mat');

lambda = 1e-2;
timeLimitSecs = 240;

% Call rbfKernel.m to precompute kappa2 up here to ensure fairness
tic
[~] = rbfKernel(TrainingX(1, :), TrainingX(2, :), TrainingX);
% [~] = rbfKernel(TrainingX(1, :), TrainingX(2, :), TrainingX, true);
toc

% Call makek_i for all training datapoints to precompute K up here to
% ensure fairness
tic
for i = 1:size(TrainingX, 1)
    [~] = makek_i(i, TrainingX);
%     [~] = makek_i(i, TrainingX, true);
end
toc


%% Full gradient descent
% stepSize = 1;
% stepSize = .5;
% stepSize = .2;
stepSize = .1;
% stepSize = .01;
start_FGD = tic;
[bestW_FGD, ~, Jw_FGD, timeByIter_FGD, ~] = ...
    gradDesc(TrainingX, TrainingY, lambda, stepSize);
time_FGD = toc(start_FGD);
testAcc_FGD = testAccuracy(bestW_FGD, TrainingX, TestX, TestY);

%% Stochastic gradient descent, p = 100
% stepSize = .1;
% stepSize = .05;
stepSize = .02;
% stepSize = .01;
% stepSize = .001;
start_SGDp100 = tic;
[bestW_SGDp100, ~, Jw_SGDp100, timeByIter_SGDp100 ,~] = ...
    gradDesc(TrainingX, TrainingY, lambda, stepSize, 100, timeLimitSecs);
time_SGDp100 = toc(start_SGDp100);
% timePerIter_SGDp100 = timeLimitSecs / length(Jw_SGDp100);
testAcc_SGDp100 = testAccuracy(bestW_SGDp100, TrainingX, TestX, TestY);

%% Stochastic gradient descent, p = 1
% stepSize = .01;
stepSize = .005;
% stepSize = .002;
% stepSize = .001;
start_SGDp1 = tic;
[bestW_SGDp1, ~, Jw_SGDp1, timeByIter_SGDp1, ~] = ...
    gradDesc(TrainingX, TrainingY, lambda, stepSize, 1, timeLimitSecs);
time_SGDp1 = toc(start_SGDp1);
% timePerIter_SGDp1 = timeLimitSecs / length(Jw_SGDp1);
testAcc_SGDp1 = testAccuracy(bestW_SGDp1, TrainingX, TestX, TestY);

%% Make plots
timeToPlotUntil = 10;
iterAtTime_FGD = find(timeByIter_FGD >= timeToPlotUntil, 1);
iterAtTime_SGDp100 = find(timeByIter_SGDp100 >= timeToPlotUntil, 1);
iterAtTime_SGDp1 = find(timeByIter_SGDp1 >= timeToPlotUntil, 1);

figure; 
plot(timeByIter_FGD(1:iterAtTime_FGD), Jw_FGD(1:iterAtTime_FGD));
title('J(w) vs. time (seconds), full gradient descent');

figure; 
plot(timeByIter_SGDp100(1:iterAtTime_SGDp100), ...
    Jw_SGDp100(1:iterAtTime_SGDp100));
title('J(w) vs. time (seconds), SGD with p = 100');

figure; 
plot(timeByIter_SGDp1(1:iterAtTime_SGDp1), Jw_SGDp1(1:iterAtTime_SGDp1));
title('J(w) vs. time (seconds), SGD with p = 1');

figure;
plot(timeByIter_FGD(1:iterAtTime_FGD), Jw_FGD(1:iterAtTime_FGD), 'r');
hold on;
plot(timeByIter_SGDp100(1:iterAtTime_SGDp100), ...
    Jw_SGDp100(1:iterAtTime_SGDp100), 'g');
plot(timeByIter_SGDp1(1:iterAtTime_SGDp1), ...
    Jw_SGDp1(1:iterAtTime_SGDp1), 'b');
title(['J(w) vs. time (seconds). Red: full GD. Green: SGD, p = 100. ' ...
    'Blue: SGD, p = 1.']);
