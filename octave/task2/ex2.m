%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.

fprintf(['Визуализируем данные, знак + означает примеры с y = 1, знак o ' ...
         'обозначает примеры с y = 0.\n']);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('Нажмитие Enter, чтобы продолжить.\n');
pause;


%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunction.m

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Стоимость для начального значения theta (нулевого): %f\n', cost);
fprintf('Ожидаемая стоимость (примерно): 0.693\n');
fprintf('Градиент для начального значения theta (нулевого): \n');
fprintf(' %f \n', grad);
fprintf('Ожидаемые значения градиента (примерно):\n -0.1000\n -12.0092\n -11.2628\n');

% Compute and display cost and gradient with non-zero theta
test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);

fprintf('\nСтоимость для тестового значения: %f\n', cost);
fprintf('Ожидаемая стоимость (примерно): 0.218\n');
fprintf('Градиент для тестового значения: \n');
fprintf(' %f \n', grad);
fprintf('Ожидаемые значения градиента (примерно):\n 0.043\n 2.566\n 2.647\n');

fprintf('Нажмитие Enter, чтобы продолжить.\n');
pause;


%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Стоимость для theta, найденная fminunc: %f\n', cost);
fprintf('Ожидаемая стоимость (примерно): 0.203\n');
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('Ожидаемое значение theta (примерно):\n');
fprintf(' -25.161\n 0.206\n 0.201\n');

% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('Нажмитие Enter, чтобы продолжить.\n');
pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

prob = sigmoid([1 45 85] * theta);
fprintf(['Для студента с баллами 45 и 85 мы прогнозируем вероятность ' ...
         'поступления %f\n'], prob);
fprintf('Ожидаемое значение: 0.775 +/- 0.002\n\n');

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Точность на обучающей выборке: %f\n', mean(double(p == y)) * 100);
fprintf('Ожидаемая точность (примерно): 89.0\n');
fprintf('\n');


