function los = logOfSigmoid(v)
% This function exists to prevent very small vs from being rounded to 0
% making log produce -Inf.

los = log(sigmoid(v));
if isinf(los)
    los = v;
end
