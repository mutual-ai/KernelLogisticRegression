function acc = testAccuracy(w, TrainX, TestX, TestY)

correct = 0;
for i = 1:size(TestX, 1)
    x_i = TestX(i, :);
    y_i = TestY(i, :);
    
    % Check Piazza question to make sure ks should be computed from
    % training data
    k_i = makek_i(-1, TrainX, false, x_i);
    probability = sigmoid(w' * k_i);
    
    if probability > .5
        guess = 1;
    else
        guess = -1;
    end
    
    if guess == y_i
        correct = correct + 1;
    end
end
acc = correct / size(TestX, 1);
