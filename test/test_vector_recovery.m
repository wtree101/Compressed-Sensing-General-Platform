%% Simple Test for Vector Recovery Framework
clear; clc;

fprintf('Testing Vector Recovery Framework...\n');

% Set up test parameters
d1 = 50; % Vector dimension
sparsity = 5; % Sparsity level
m = 25; % Number of measurements (should be >= 2*sparsity for exact recovery)
kappa = 2;
lambda = 0.001;

% Test parameters
params.d1 = d1;
params.sparsity = sparsity;
params.m = m;
params.kappa = kappa;
params.lambda = lambda;
params.trial_num = 1;
params.verbose = true;
params.T = 100;
params.init_scale = 1e-3;
params.problem_flag = 0; % Gaussian sensing
params.lambda = lambda;
params.apply_thresholding = true;
params.alg = @solve_GD_vec;
params.init = @init_random_vector;

% Generate ground truth sparse vector
xstar = zeros(d1, 1);
support = randperm(d1, sparsity);
xstar(support) = randn(sparsity, 1);
xstar = xstar / norm(xstar);
params.xstar = xstar;

fprintf('Ground truth sparsity: %d\n', nnz(xstar));
fprintf('Ground truth norm: %.3f\n', norm(xstar));

% Run single trial
fprintf('\nRunning single trial...\n');
try
    [Error_Stand, Error_function] = onetrial_vec(params);
    
    fprintf('Initial error: %.3e\n', Error_Stand(1));
    fprintf('Final error: %.3e\n', Error_Stand(end));
    fprintf('Final function error: %.3e\n', Error_function(end));
    
    if Error_Stand(end) < 1e-3
        fprintf('✓ Recovery successful!\n');
    else
        fprintf('✗ Recovery failed\n');
    end
    
catch ME
    fprintf('✗ Test failed: %s\n', ME.message);
    return;
end

% Test multiple trials
fprintf('\nRunning multiple trials...\n');
params.trial_num = 5;
try
    [err_list, p_list, err_list_f] = multipletrial(params);
    
    fprintf('Success probability: %.2f\n', p_list);
    fprintf('Mean final error: %.3e\n', mean(err_list));
    fprintf('Mean function error: %.3e\n', mean(err_list_f));
    
catch ME
    fprintf('✗ Multiple trial test failed: %s\n', ME.message);
    return;
end

fprintf('\n=== Vector Recovery Framework Test Completed ===\n');
