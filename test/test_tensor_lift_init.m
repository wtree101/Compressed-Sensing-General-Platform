% test_tensor_lift_init.m
% Concise test for tensor lift initialization

clear; clc;
addpath('../Initialization_groundtruth');
addpath('../utilities');
addpath('../solver');

fprintf('=== Testing Tensor Lift Initialization ===\n\n');

%% Setup problem
d = 20;          % Matrix dimension
r = 1;           % Rank
m =  200;     % Number of measurements
T_power = 10;    % Tensor PGD iterations

% Generate ground truth: X = UU^T
U_true = randn(d, r);
Xstar = U_true * U_true';
Xstar = Xstar / norm(Xstar, 'fro');

fprintf('Problem size: d=%d, r=%d, m=%d\n', d, r, m);
fprintf('Ground truth: ||X||_F=%.4f, rank=%d\n\n', norm(Xstar,'fro'), rank(Xstar));

%% Create measurement operator
A = randn(m, d*d) / sqrt(m);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d, d]);

% Generate measurements
y = abs(operator.A(Xstar));

fprintf('Measurements: range=[%.4f, %.4f]\n\n', min(y), max(y));

%% Test tensor lift initialization
params = struct();
params.T_power = T_power;
params.r = r;
params.Xstar = Xstar;
params.verbose = true;

tic;
[X0, U0, history] = initialize_tensor_lift(y, operator, d, d, params);
elapsed = toc;

%% Display results
fprintf('\n=== Results ===\n');
fprintf('Initialization time: %.3f seconds\n', elapsed);
fprintf('Final matrix error: %.6e\n', history.final_error);
fprintf('Final tensor error: %.6e\n', history.tensor_errors(end));
fprintf('Initialized rank: %d (target: %d)\n', rank(X0, 1e-6), r);

%% Plot convergence
figure('Position', [100, 100, 800, 400]);

subplot(1,2,1);
semilogy(1:T_power, history.tensor_errors, 'b-o', 'LineWidth', 2);
grid on;
xlabel('Iteration');
ylabel('Tensor Error');
title('Tensor Error Convergence');

subplot(1,2,2);
semilogy(1:T_power, history.loss_function, 'r-s', 'LineWidth', 2);
grid on;
xlabel('Iteration');
ylabel('Loss Function');
title('Loss Function Convergence');

sgtitle(sprintf('Tensor Lift Initialization (d=%d, r=%d, m=%d, T=%d)', d, r, m, T_power));

%% Success check
if history.final_error < 0.5
    fprintf('\n✓ Test PASSED: Final error < 0.5\n');
else
    fprintf('\n✗ Test FAILED: Final error >= 0.5\n');
end
