%%%%%%%%%% Simple Test Script for Sparse Vector Recovery
% Quick test with fixed parameters for debugging and validation

clear; clc; close all;

fprintf('=== Simple Vector Recovery Test Script ===\n');

%% Fixed Test Parameters
% Small problem size for quick testing
d1 = 1000;         % Vector dimension
m = 200;          % Number of measurements
sparsity = 10;    % Sparsity level to recover
sparsity_star = 10; % True sparsity
kappa = 2;       % Signal strength parameter

% Small number of iterations for quick test
T = 40000;
trial_num = 1;   % Single trial for quick test
verbose = 1;     % Show plots

% Test parameters
lambda = 1e-3;   % Small regularization
problem_flag = 0; % Standard Gaussian sensing

fprintf('Test Configuration:\n');
fprintf('  Vector dimension: %d\n', d1);
fprintf('  Measurements: %d\n', m);
fprintf('  Sparsity: %d (true: %d)\n', sparsity, sparsity_star);
fprintf('  Iterations: %d\n', T);
fprintf('  Lambda: %.4f\n', lambda);

%% Generate Ground Truth
% Create sparse ground truth vector
xstar = zeros(d1, 1);
support = randperm(d1, sparsity_star);
xstar(support) = randn(sparsity_star, 1);
xstar = xstar / norm(xstar); % Normalize

fprintf('Ground truth:\n');
fprintf('  Support size: %d\n', nnz(xstar));
fprintf('  Norm: %.3f\n', norm(xstar));

%% Setup and Run Test
% Setup experiment parameters
params = struct();
params.d1 = d1;
params.m = m;
params.sparsity = sparsity;
params.kappa = kappa;
params.lambda = lambda * min(abs(xstar(xstar ~= 0)));
params.T = T;
params.trial_num = trial_num;
params.verbose = verbose;
params.problem_flag = problem_flag;
params.init_scale = 0; % 0 initialization for compressed sensing
params.xstar = xstar;
params.mu = 1;

% Set solver and initialization
params.alg = @solve_GD_vec;
params.init = @init_random_vector;
params.apply_thresholding = (lambda > 0); % Apply soft thresholding if lambda > 0
fprintf('\n--- Running Vector Recovery ---\n');
params.nonlinear_func = @(z) abs(z); % Phase retrieval nonlinearity

% Run single trial
tic;
[Error_Stand, Error_function] = onetrial_vec(params);
elapsed_time = toc;

fprintf('Results:\n');
fprintf('  Initial error: %.4e\n', Error_Stand(1));
fprintf('  Final error: %.4e\n', Error_Stand(end));
fprintf('  Function error: %.4e\n', Error_function(end));
fprintf('  Computation time: %.3f seconds\n', elapsed_time);

% Check convergence
if Error_Stand(end) < 1e-2
    fprintf('  Status: CONVERGED ✓\n');
else
    fprintf('  Status: Not converged\n');
end

%% Plot Results
figure('Position', [100, 100, 800, 300]);

% Plot convergence curve
subplot(1, 2, 1);
semilogy(1:T, Error_Stand, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Relative Error');
title('Vector Recovery Convergence');
grid on;

% Plot function error
subplot(1, 2, 2);
semilogy(1:T, Error_function, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Function Error');
title('Function Error');
grid on;

%% Save Test Results
save('simple_test_vec_results.mat', 'Error_Stand', 'Error_function', ...
     'T', 'd1', 'm', 'sparsity', 'sparsity_star', 'xstar', 'elapsed_time');

fprintf('\n=== Vector Recovery Test Complete ===\n');
fprintf('Results saved to: simple_test_vec_results.mat\n');

% Summary
fprintf('\nSummary:\n');
fprintf('  Problem: %d-sparse vector in R^%d from %d measurements\n', sparsity_star, d1, m);
fprintf('  Measurement ratio: %.2f (theory needs ~%.1f)\n', m/sparsity_star, 2*log(d1/sparsity_star));
fprintf('  Final recovery error: %.2e\n', Error_Stand(end));
