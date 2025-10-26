%%%%%%%%%% Simple Test Script for Low-rank Matrix Recovery with PGD
% Quick test with fixed parameters for debugging and validation

clear; clc; close all;

fprintf('=== Simple Test Script for PGD ===\n');

%% Fixed Test Parameters
% Small problem size for quick testing
d1 = 20; d2 = 20;
m = 100;        % Number of measurements
r = 2;          % Rank to recover
r_star = 2;     % True rank
kappa = 2;      % Condition number

% Small number of iterations for quick test
T = 1000;
trial_num = 1;  % Single trial for quick test
verbose = 1;    % Show plots

% Test different algorithms, focus on PGD
algorithms = [0, 1, 4]; % GD, RGD, PGD
alg_names = {'GD', 'RGD', 'PGD'};

% Test parameters
lambda = 0;           % No regularization
problem_flag = 2;     % Standard sensing

fprintf('Test Configuration:\n');
fprintf('  Matrix size: %dx%d\n', d1, d2);
fprintf('  Measurements: %d\n', m);
fprintf('  Rank: %d (true: %d)\n', r, r_star);
fprintf('  Iterations: %d\n', T);
fprintf('  Trials: %d\n', trial_num);

%% Run Tests for Each Algorithm
results = cell(length(algorithms), 1);

for alg_idx = 1:length(algorithms)
    alg_flag = algorithms(alg_idx);
    alg_name = alg_names{alg_idx};
    
    fprintf('\n--- Testing %s Algorithm ---\n', alg_name);
    
    % Setup experiment parameters
    params = setup_experiment_params(d1, d2, kappa, trial_num, verbose, ...
                                    problem_flag, alg_flag, r_star, T);
    
    % Add current experiment parameters to params
    params.lambda = lambda;
    params.m = m;
    params.r = r;
    params.kappa = kappa;
    
    % Add projection function for PGD
    if alg_flag == 4 % PGD algorithm
        params.projection = @rank_projection;
    end
    
    % Run single trial
    tic;
    try
        [Error_Stand, Error_function] = onetrial(params);
        elapsed_time = toc;
        
        % Store results
        results{alg_idx} = struct();
        results{alg_idx}.alg_name = alg_name;
        results{alg_idx}.Error_Stand = Error_Stand;
        results{alg_idx}.Error_function = Error_function;
        results{alg_idx}.final_error = Error_Stand(end);
        results{alg_idx}.time = elapsed_time;
        
        fprintf('  Final relative error: %.6f\n', Error_Stand(end));
        fprintf('  Computation time: %.3f seconds\n', elapsed_time);
        
        % Check convergence
        if Error_Stand(end) < 1e-3
            fprintf('  Status: CONVERGED ✓\n');
        else
            fprintf('  Status: Not converged\n');
        end
    catch ME
        elapsed_time = toc;
        fprintf('  Error running algorithm: %s\n', ME.message);
        results{alg_idx} = []; % Mark as failed
    end
end

%% Display Comparison
fprintf('\n=== Algorithm Comparison ===\n');
fprintf('Algorithm   Final Error   Time (s)   Status\n');
fprintf('------------------------------------------\n');

for alg_idx = 1:length(algorithms)
    res = results{alg_idx};
    if ~isempty(res)
        status = 'Failed';
        if res.final_error < 1e-3
            status = 'Pass';
        end
        fprintf('%-10s  %.4e     %.3f      %s\n', ...
                res.alg_name, res.final_error, res.time, status);
    else
        fprintf('%-10s  N/A           N/A        Error\n', alg_names{alg_idx});
    end
end

%% Plot Results
figure('Position', [100, 100, 1200, 400]);

% Plot convergence curves
subplot(1, 3, 1);
hold on;
colors = ['b', 'r', 'g', 'm', 'c', 'k'];
for alg_idx = 1:length(algorithms)
    res = results{alg_idx};
    if ~isempty(res) % Check if algorithm ran successfully
        semilogy(1:T, res.Error_Stand, colors(alg_idx), 'LineWidth', 2, ...
                 'DisplayName', res.alg_name);
    end
end
xlabel('Iteration');
ylabel('Relative Error');
title('Convergence Comparison');
legend('show');
grid on;

% Plot function error
subplot(1, 3, 2);
hold on;
for alg_idx = 1:length(algorithms)
    res = results{alg_idx};
    if ~isempty(res) % Check if algorithm ran successfully
        semilogy(1:T, res.Error_function, colors(alg_idx), 'LineWidth', 2, ...
                 'DisplayName', res.alg_name);
    end
end
xlabel('Iteration');
ylabel('Function Error');
title('Function Error');
legend('show');
grid on;

% Bar plot of final errors
subplot(1, 3, 3);
final_errors = [];
successful_algs = {};
for alg_idx = 1:length(algorithms)
    if ~isempty(results{alg_idx})
        final_errors(end+1) = results{alg_idx}.final_error;
        successful_algs{end+1} = results{alg_idx}.alg_name;
    end
end
if ~isempty(final_errors)
    bar(final_errors);
    set(gca, 'XTickLabel', successful_algs);
    ylabel('Final Relative Error');
    title('Final Error Comparison');
    set(gca, 'YScale', 'log');
    grid on;
else
    text(0.5, 0.5, 'No successful algorithms', 'HorizontalAlignment', 'center');
    title('Final Error Comparison');
end

%% Save Test Results
save('simple_test_pgd_results.mat', 'results', 'T', 'd1', 'd2', 'm', 'r', 'r_star');

fprintf('\n=== Test Complete ===\n');
fprintf('Results saved to: simple_test_pgd_results.mat\n');
fprintf('Plots displayed for visual inspection\n');


