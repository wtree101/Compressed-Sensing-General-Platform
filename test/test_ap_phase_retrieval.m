%%%%%%%%%% Test Script for AP (Alternating Projection) Phase Retrieval
% Test AP algorithm for real sparse phase retrieval problem

clear; clc; close all;

fprintf('=== AP Phase Retrieval Test (Real Signals) ===\n');

%% Problem Setup for Sparse Phase Retrieval
% Signal parameters
n = 100;                % Signal dimension (vector length)
m = 1000;                % Number of measurements (should be > 2*sparsity)
sparsity = 1;          % Number of non-zero elements (sparsity level)

% For compatibility with matrix-based solver interface
d1 = n; d2 = 1;         % Treat as n×1 matrix (column vector)

% Algorithm parameters
T = 500;                % Number of iterations
verbose = 1;            % Show progress

fprintf('Problem Configuration:\n');
fprintf('  Signal length: %d\n', n);
fprintf('  Measurements: %d\n', m);
fprintf('  Sparsity level: %d\n', sparsity);
fprintf('  Iterations: %d\n', T);

%% Generate Sparse Ground Truth Signal
% Create sparse vector with random support
support_idx = randperm(n, sparsity);
x_true = zeros(n, 1);

% Generate random real values for sparse entries
sparse_values = randn(sparsity, 1);
x_true(support_idx) = sparse_values;

% Reshape for matrix interface compatibility
Xstar = reshape(x_true, [d1, d2]);

fprintf('Ground truth sparsity: %d (should be %d)\n', nnz(x_true), sparsity);

%% Generate Random Gaussian Measurements
% Create random real measurement matrix
A = randn(m, n);

% Define operators
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(z) reshape(A' * z, [d1, d2]);

% Generate measurements (magnitude-only measurements for phase retrieval)
z_true = operator.A(Xstar) / sqrt(m);
y = abs(z_true); % Phase retrieval: only magnitude measurements available

fprintf('Measurement range: [%.3f, %.3f]\n', min(y), max(y));

%% Test Different Initialization Methods
init_methods = {'random', 'power'};
num_methods = length(init_methods);

results = cell(num_methods, 1);

for init_idx = 1:num_methods
    init_method = init_methods{init_idx};
    fprintf('\n--- Testing %s initialization ---\n', init_method);
    
    %% Initialize
    switch init_method
        case 'random'
            Xl_init = reshape(randn(n, 1), [d1, d2])*0.1;
        case 'power'
            T_power = 50; % Number of power iterations
            [Xl_init, ~] = initialize_power_method(y, operator, d1, d2, T_power, false);
            % Scale the initialization
            Xl_init = Xl_init * norm(y, 'fro') / norm(operator.A(Xl_init), 'fro');
    end
    
    %% Setup AP Parameters
    params = struct();
    params.T = T;
    params.Xstar = Xstar;
    params.nonlinear_func = @(z) abs(z); % Phase retrieval: observe magnitudes only
    params.sparsity = sparsity;
    
    % Sparsity projection function
    params.projection = @sparsity_projection;
    
    % PGD parameters for subproblem
    params.mu = 0.1;
    params.lambda = 0;
    
    %% Run AP Algorithm
    tic;
    [solver_output, Xl_final] = solve_AP(Xl_init, [], y, operator, d1, d2, [], [], params);
    Error_Stand = solver_output.Error_Stand;
    Error_function = solver_output.Error_function;
    elapsed_time = toc;
    
    %% Store Results
    results{init_idx} = struct();
    results{init_idx}.init_method = init_method;
    results{init_idx}.Error_Stand = Error_Stand;
    results{init_idx}.Error_function = Error_function;
    results{init_idx}.Xl_final = Xl_final;
    results{init_idx}.final_error = Error_Stand(end);
    results{init_idx}.time = elapsed_time;
    
    % Check recovery quality
    recovered_sparsity = nnz(abs(Xl_final) > 0.1 * max(abs(Xl_final(:))));
    support_recovery = length(intersect(support_idx, find(abs(Xl_final(:)) > 0.1 * max(abs(Xl_final(:))))));
    
    fprintf('  Final relative error: %.6f\n', Error_Stand(end));
    fprintf('  Recovered sparsity: %d (true: %d)\n', recovered_sparsity, sparsity);
    fprintf('  Support recovery: %d/%d (%.1f%%)\n', support_recovery, sparsity, 100*support_recovery/sparsity);
    fprintf('  Computation time: %.3f seconds\n', elapsed_time);
    
    % Store additional metrics
    results{init_idx}.recovered_sparsity = recovered_sparsity;
    results{init_idx}.support_recovery_rate = support_recovery / sparsity;
    
    % Check convergence
    if Error_Stand(end) < 1e-2 && support_recovery >= 0.8 * sparsity
        fprintf('  Status: RECOVERED ✓\n');
        results{init_idx}.status = 'Success';
    else
        fprintf('  Status: Not recovered\n');
        results{init_idx}.status = 'Failed';
    end
end

%% Display Comparison
fprintf('\n=== Initialization Method Comparison ===\n');
fprintf('Method      Final Error   Support Rec.   Time (s)   Status\n');
fprintf('--------------------------------------------------------\n');

for init_idx = 1:num_methods
    res = results{init_idx};
    fprintf('%-10s  %.4e     %.1f%%         %.3f      %s\n', ...
            res.init_method, res.final_error, 100*res.support_recovery_rate, ...
            res.time, res.status);
end

%% Detailed Analysis for Best Result
[~, best_idx] = min(cellfun(@(x) x.final_error, results));
best_result = results{best_idx};

fprintf('\n=== Best Result Analysis (%s initialization) ===\n', best_result.init_method);

% Compare true and recovered signals
Xl_best = best_result.Xl_final;
fprintf('Signal comparison:\n');
fprintf('  True signal norm: %.6f\n', norm(Xstar, 'fro'));
fprintf('  Recovered signal norm: %.6f\n', norm(Xl_best, 'fro'));
fprintf('  Relative error: %.6f\n', norm(Xl_best - Xstar, 'fro') / norm(Xstar, 'fro'));

% For real signals, no phase alignment needed
fprintf('  Direct comparison (real signals only)\n');

%% Plot Results
figure('Position', [100, 100, 1500, 1000]);

% Plot 1: Convergence curves
subplot(2, 3, 1);
hold on;
colors = ['b', 'r', 'g', 'm', 'c'];
for init_idx = 1:num_methods
    res = results{init_idx};
    semilogy(1:T, res.Error_Stand, colors(init_idx), 'LineWidth', 2, ...
             'DisplayName', res.init_method);
end
xlabel('Iteration');
ylabel('Relative Error');
title('Convergence Comparison');
legend('show');
grid on;

% Plot 2: Function error
subplot(2, 3, 2);
hold on;
for init_idx = 1:num_methods
    res = results{init_idx};
    semilogy(1:T, res.Error_function, colors(init_idx), 'LineWidth', 2, ...
             'DisplayName', res.init_method);
end
xlabel('Iteration');
ylabel('Function Error');
title('Function Error');
legend('show');
grid on;

% Plot 3: Support recovery rates
subplot(2, 3, 3);
support_rates = cellfun(@(x) 100*x.support_recovery_rate, results);
bar(support_rates);
set(gca, 'XTickLabel', init_methods);
ylabel('Support Recovery Rate (%)');
title('Support Recovery');
ylim([0, 105]);
grid on;

% Plot 4: True signal
subplot(2, 3, 4);
stem(1:n, x_true, 'b', 'LineWidth', 2);
title('True Sparse Vector');
xlabel('Index');
ylabel('Value');
grid on;

% Plot 5: Best recovered signal
subplot(2, 3, 5);
stem(1:n, Xl_best(:), 'r', 'LineWidth', 2);
title(['Recovered Vector (', best_result.init_method, ')']);
xlabel('Index');
ylabel('Value');
grid on;

% Plot 6: Support comparison
subplot(2, 3, 6);
hold on;
true_support = find(abs(x_true) > 1e-10);
recovered_support = find(abs(Xl_best(:)) > 0.1 * max(abs(Xl_best(:))));
stem(true_support, ones(length(true_support), 1), 'bo', 'MarkerSize', 8, 'DisplayName', 'True');
stem(recovered_support, 0.5*ones(length(recovered_support), 1), 'ro', 'MarkerSize', 8, 'DisplayName', 'Recovered');
xlabel('Index');
ylabel('Support Indicator');
title('Support Comparison');
legend('show');
ylim([0, 1.2]);
grid on;

%% Save Results
save('ap_phase_retrieval_real_test_results.mat', 'results', 'x_true', 'Xstar', 'y', 'params', ...
     'sparsity', 'n', 'm', 'T');

fprintf('\n=== Test Complete ===\n');
fprintf('Results saved to: ap_phase_retrieval_real_test_results.mat\n');

%% Helper Function: Sparsity Projection
function X_proj = project_sparse(X, s)
    % Project onto the set of s-sparse signals
    % Keep the s largest magnitude entries, set others to zero
    
    X_vec = X(:);
    [~, idx] = sort(abs(X_vec), 'descend');
    
    X_proj_vec = zeros(size(X_vec));
    X_proj_vec(idx(1:s)) = X_vec(idx(1:s));
    
    X_proj = reshape(X_proj_vec, size(X));
end
