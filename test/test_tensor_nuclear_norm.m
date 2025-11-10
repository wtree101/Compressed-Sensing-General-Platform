%% Test Tensor Nuclear Norm Minimization Solver - Preconditioner Evaluation
% This script tests the solve_tensor_nuclear_norm function for phase retrieval
% using tensor nuclear norm minimization (convex relaxation)
%
% Test objectives:
%   1. Evaluate PCG preconditioner effectiveness
%   2. Compare with and without Jacobi preconditioner
%   3. Test different PCG iteration counts
%
% New default strategy: rho=0.1, over_relax=1.0, pcg_maxit=5, no preconditioner

clear; clc;

fprintf('=== Test Tensor Nuclear Norm Minimization - Preconditioner Evaluation ===\n\n');

%% Test Parameters
d = 20;              % Dimension (start small for testing)
r = 1;               % Rank of ground truth matrix
m = 192;             % Number of measurements (m >= d^2 for recovery)

% Test configurations to evaluate preconditioner impact
% Focus on: preconditioner on/off, PCG iterations
test_configs = {
    % [rho, alpha, pcg_tol, pcg_maxit, use_precond, label]
    {0.1, 1.0, 1e-1, 0,  false, 'Baseline (no precond, maxit=1)'},
    {0.1, 1.0, 1e-2, 5,  true,  'With Jacobi precond (maxit=5)'},
    {0.1, 1.0, 1e-2, 10, false, 'No precond (maxit=10)'},
    {0.1, 1.0, 1e-2, 10, true,  'With precond (maxit=10)'},
    {0.1, 1.0, 1e-2, 20, false, 'No precond (maxit=20)'},
    {0.1, 1.0, 1e-2, 20, true,  'With precond (maxit=20)'},
    {0.1, 1.5, 1e-2, 5,  false, 'Over-relax α=1.5, no precond'},
    {0.1, 1.5, 1e-2, 5,  true,  'Over-relax α=1.5, with precond'}
};
n_configs = length(test_configs);

fprintf('Test Configuration:\n');
fprintf('  Dimension d: %d\n', d);
fprintf('  Rank r: %d\n', r);
fprintf('  Measurements m: %d\n', m);
fprintf('  Measurement ratio m/d²: %.2f\n', m / d^2);
fprintf('  Configurations to test: %d\n\n', n_configs);

%% Generate Ground Truth (symmetric low-rank matrix)
fprintf('Generating ground truth...\n');
rng(42);  % For reproducibility

U_true = randn(d, r);
U_true = orth(U_true);  % Orthonormalize
Xstar = abs(U_true) * abs(U_true)';  % Symmetric rank-r matrix
Xstar = Xstar / norm(Xstar, 'fro');

fprintf('  Ground truth: %dx%d symmetric matrix\n', d, d);
fprintf('  Rank: %d\n', rank(Xstar, 1e-6));
fprintf('  Norm: %.6f\n\n', norm(Xstar, 'fro'));

%% Create Measurement Operator
fprintf('Creating measurement operator...\n');

% Random Gaussian measurements
A_cells = cell(m, 1);
for i = 1:m
    Ai = randn(d, d);
    A_cells{i} = (Ai + Ai') / 2;  % Symmetrize for consistency
end

operator = struct();
operator.A_cells = A_cells;

fprintf('  Created %d measurement matrices\n\n', m);

%% Generate Measurements (phase retrieval)
fprintf('Generating measurements...\n');

% Compute y_i = <A_i ⊗ A_i, X ⊗ X>
y = zeros(m, 1);
for i = 1:m
    % y_i = <A_i, X>² for phase retrieval model
    y(i) = abs(trace(A_cells{i}' * Xstar))^2 / sqrt(m);
end

fprintf('  Measurements generated\n');
fprintf('  Measurement vector norm: %.6f\n\n', norm(y));

%% Solve using Tensor Nuclear Norm Minimization - Parameter Tuning
fprintf('=== Running Tensor Nuclear Norm Solver - Parameter Tuning ===\n\n');

% Fixed parameters
max_iter = 100;
lambda = 1.0;       % Regularization parameter
verbose = 0;        % Reduced verbosity for sweep

fprintf('Fixed parameters:\n');
fprintf('  lambda: %.2f, max_iter: %d\n\n', lambda, max_iter);

% Storage for all results
all_results = cell(n_configs, 1);

% Test each configuration
fprintf('Testing %d configurations...\n', n_configs);
fprintf('%-40s | %-10s | %-10s | %-8s | %-10s\n', ...
        'Config', 'Recon Err', 'Rel Err', 'Iter', 'Time(s)');
fprintf('%s\n', repmat('-', 1, 85));

for c_idx = 1:n_configs
    cfg = test_configs{c_idx};
    cfg_rho = cfg{1};
    cfg_alpha = cfg{2};
    cfg_pcg_tol = cfg{3};
    cfg_pcg_maxit = cfg{4};
    cfg_use_precond = cfg{5};
    cfg_label = cfg{6};
    
    % Call solver with current configuration
    tic;
    [X_recovered, info] = solve_tensor_nuclear_norm(operator, y, d, ...
        'max_iter', max_iter, ...
        'lambda', lambda, ...
        'rho', cfg_rho, ...
        'verbose', verbose, ...
        'X_true', Xstar, ...
        'over_relax', cfg_alpha, ...
        'pcg_tol', cfg_pcg_tol, ...
        'pcg_maxit', cfg_pcg_maxit, ...
        'use_pcg_precond', cfg_use_precond);
    elapsed_time = toc;
    
    % Compute reconstruction error
    [recon_error, X_recovered_aligned] = rectify_sign_ambiguity(X_recovered, Xstar);
    rel_error = norm(X_recovered_aligned - Xstar, 'fro') / norm(Xstar, 'fro');
    
    % Store results
    results.config_label{c_idx} = cfg_label;
    results.rho(c_idx) = cfg_rho;
    results.alpha(c_idx) = cfg_alpha;
    results.pcg_tol(c_idx) = cfg_pcg_tol;
    results.pcg_maxit(c_idx) = cfg_pcg_maxit;
    results.use_precond(c_idx) = cfg_use_precond;
    results.recon_error(c_idx) = recon_error;
    results.rel_error(c_idx) = rel_error;
    results.converged(c_idx) = info.converged;
    results.iterations(c_idx) = info.iter;
    results.time(c_idx) = elapsed_time;
    results.final_obj(c_idx) = info.obj_values(end);
    results.final_constraint(c_idx) = info.constraint_violations(end);
    results.final_rho(c_idx) = info.final_rho;
    results.rank(c_idx) = rank(X_recovered, 1e-6);
    results.T_norm(c_idx) = norm(X_recovered, 'fro');
    results.X_recovered{c_idx} = X_recovered_aligned;
    results.info{c_idx} = info;
    
    % Print summary
    fprintf('%-30s | %-10.4f | %-10.4e | %-10.4e | %-8d | %-10.4f | %-10s\n', ...
            cfg_label, cfg_alpha, recon_error, rel_error, info.iter, elapsed_time, mat2str(info.converged));
end

fprintf('%s\n', repmat('-', 1, 120));

%% Find Best Configuration
fprintf('\n\n=== Configuration Comparison ===\n\n');

[best_error, best_idx] = min(results.recon_error);

fprintf('%-30s | %-12s | %-12s | %-10s\n', 'Configuration', 'Recon Error', 'Iterations', 'Time (s)');
fprintf('%s\n', repmat('-', 1, 75));

for idx = 1:n_configs
    fprintf('%-30s | %-12.4e | %-12d | %-10.4f\n', ...
            results.config_label{idx}, results.recon_error(idx), results.iterations(idx), results.time(idx));
end

fprintf('%s\n\n', repmat('-', 1, 75));

fprintf('=== Best Configuration ===\n');
fprintf('Label: %s\n', results.config_label{best_idx});
fprintf('  Rho: %.2f\n', results.rho(best_idx));
fprintf('  Over-relax α: %.2f\n', results.alpha(best_idx));
fprintf('  PCG tolerance: %.1e\n', results.pcg_tol(best_idx));
fprintf('  PCG max iter: %d\n', results.pcg_maxit(best_idx));
fprintf('  PCG preconditioner: %s\n', mat2str(results.use_precond(best_idx)));
fprintf('  Reconstruction error: %.6e\n', best_error);
fprintf('  Relative error: %.6e\n', results.rel_error(best_idx));
fprintf('  Iterations: %d\n', results.iterations(best_idx));
fprintf('  Time: %.4f seconds\n', results.time(best_idx));
fprintf('  Converged: %s\n', mat2str(results.converged(best_idx)));
fprintf('  Recovered rank: %d\n\n', results.rank(best_idx));

% Select best result for detailed analysis
X_recovered = results.X_recovered{best_idx};
info = results.info{best_idx};

%% Evaluate Reconstruction (Best Result)
fprintf('=== Reconstruction Evaluation (Best Result) ===\n');

fprintf('Reconstruction error: %.6e\n', best_error);
fprintf('Recovered matrix rank: %d\n', rank(X_recovered, 1e-6));
fprintf('Recovered matrix norm: %.6f\n', norm(X_recovered, 'fro'));

% Check matrix symmetry (X should be symmetric)
matrix_symmetry_error = norm(X_recovered - X_recovered', 'fro');
fprintf('Matrix symmetry error: %.6e\n', matrix_symmetry_error);

% Relative error
fprintf('Relative error: %.6e\n', results.rel_error(best_idx));

%% Preconditioner Analysis Plots
fprintf('\n=== Generating Preconditioner Analysis Plots ===\n');

figure('Position', [100, 100, 1600, 1000]);

% Plot 1: Reconstruction error for all configurations
subplot(2, 3, 1);
bar(1:n_configs, results.recon_error);
set(gca, 'XTick', 1:n_configs, 'XTickLabel', results.config_label, 'XTickLabelRotation', 45);
ylabel('Reconstruction Error');
title('Reconstruction Error by Configuration');
grid on;

% Plot 2: Preconditioner comparison (grouped by PCG maxit)
subplot(2, 3, 2);
pcg_maxits_unique = unique(results.pcg_maxit);
n_groups = length(pcg_maxits_unique);
no_precond_errors = zeros(n_groups, 1);
with_precond_errors = zeros(n_groups, 1);
for g = 1:n_groups
    maxit_val = pcg_maxits_unique(g);
    % Find configs with this maxit value
    idx_no = find(results.pcg_maxit == maxit_val & ~results.use_precond & results.alpha == 1.0, 1);
    idx_with = find(results.pcg_maxit == maxit_val & results.use_precond & results.alpha == 1.0, 1);
    if ~isempty(idx_no) && ~isempty(idx_with)
        no_precond_errors(g) = results.recon_error(idx_no);
        with_precond_errors(g) = results.recon_error(idx_with);
    end
end
bar_data = [no_precond_errors, with_precond_errors];
bar(bar_data);
set(gca, 'XTickLabel', arrayfun(@num2str, pcg_maxits_unique, 'UniformOutput', false));
xlabel('PCG max iterations');
ylabel('Reconstruction Error');
title('Preconditioner Impact');
legend('No Precond', 'With Jacobi Precond', 'Location', 'best');
grid on;

% Plot 3: Iterations by configuration
subplot(2, 3, 3);
bar(1:n_configs, results.iterations);
set(gca, 'XTick', 1:n_configs, 'XTickLabel', results.config_label, 'XTickLabelRotation', 45);
ylabel('Iterations');
title('Convergence Speed');
grid on;

% Plot 4: Time by configuration
subplot(2, 3, 4);
bar(1:n_configs, results.time);
set(gca, 'XTick', 1:n_configs, 'XTickLabel', results.config_label, 'XTickLabelRotation', 45);
ylabel('Time (s)');
title('Computation Time');
grid on;

% Plot 5: Effect of rho
subplot(2, 3, 5);
rho_values = results.rho;
[unique_rho, ~, rho_idx] = unique(rho_values);
rho_errors = zeros(size(unique_rho));
for i = 1:length(unique_rho)
    rho_errors(i) = mean(results.recon_error(rho_idx == i));
end
plot(unique_rho, rho_errors, 'mo-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Penalty Parameter ρ');
ylabel('Avg Reconstruction Error');
title('Effect of Rho');
grid on;

% Plot 6: PCG tolerance effect
subplot(2, 3, 6);
pcg_tols = results.pcg_tol;
pcg_maxits = results.pcg_maxit;
scatter(pcg_tols, results.recon_error, 100, results.time, 'filled');
xlabel('PCG Tolerance');
ylabel('Reconstruction Error');
title('PCG Settings Impact');
colorbar;
ylabel(colorbar, 'Time (s)');
grid on;
set(gca, 'XScale', 'log');

sgtitle('PCG Preconditioner Evaluation: Tensor Nuclear Norm Minimization');

%% Tensor Symmetry Analysis
fprintf('\n=== Tensor Symmetry Analysis ===\n');
fprintf('Note: Tensor symmetry analysis skipped (not implemented in current solver)\n\n');

if false && ~isempty(T_tucker) && isa(T_tucker, 'TuckerTensor') && length(T_tucker.U) >= 4
    fprintf('Tucker tensor structure:\n');
    fprintf('  Order: %d\n', T_tucker.order);
    fprintf('  Dimensions: [%s]\n', num2str(T_tucker.dims));
    fprintf('  Tucker ranks: [%s]\n', num2str(T_tucker.tucker_ranks));
    fprintf('  Symmetric flag: %s\n\n', mat2str(T_tucker.is_symmetric));
    
    % Compute pairwise factor differences
    fprintf('Factor matrix comparisons:\n');
    sym_err_12 = norm(T_tucker.U{1} - T_tucker.U{2}, 'fro');
    sym_err_13 = norm(T_tucker.U{1} - T_tucker.U{3}, 'fro');
    sym_err_14 = norm(T_tucker.U{1} - T_tucker.U{4}, 'fro');
    sym_err_23 = norm(T_tucker.U{2} - T_tucker.U{3}, 'fro');
    sym_err_24 = norm(T_tucker.U{2} - T_tucker.U{4}, 'fro');
    sym_err_34 = norm(T_tucker.U{3} - T_tucker.U{4}, 'fro');
    
    fprintf('  ||U{1} - U{2}||_F = %.6e\n', sym_err_12);
    fprintf('  ||U{1} - U{3}||_F = %.6e\n', sym_err_13);
    fprintf('  ||U{1} - U{4}||_F = %.6e\n', sym_err_14);
    fprintf('  ||U{2} - U{3}||_F = %.6e\n', sym_err_23);
    fprintf('  ||U{2} - U{4}||_F = %.6e\n', sym_err_24);
    fprintf('  ||U{3} - U{4}||_F = %.6e\n', sym_err_34);
    
    % Average and max symmetry error
    all_sym_errors = [sym_err_12, sym_err_13, sym_err_14, sym_err_23, sym_err_24, sym_err_34];
    avg_sym_error = mean(all_sym_errors);
    max_sym_error = max(all_sym_errors);
    
    fprintf('\n  Average symmetry error: %.6e\n', avg_sym_error);
    fprintf('  Maximum symmetry error: %.6e\n', max_sym_error);
    
    % Check if factors are approximately equal
    symmetry_tolerance = 1e-3;
    is_symmetric = max_sym_error < symmetry_tolerance;
    fprintf('\n  Symmetric (tol=%.0e): %s\n', symmetry_tolerance, mat2str(is_symmetric));
    
    if is_symmetric
        fprintf('  ✓ Tensor factors are symmetric!\n');
    else
        fprintf('  ✗ Tensor factors are NOT symmetric (error > tolerance)\n');
    end
    
    % Visualize factor matrices
    fprintf('\nFactor matrix statistics:\n');
    for i = 1:4
        fprintf('  U{%d}: size=%s, norm=%.6f, rank=%d\n', ...
                i, mat2str(size(T_tucker.U{i})), ...
                norm(T_tucker.U{i}, 'fro'), rank(T_tucker.U{i}));
    end
else
    fprintf('Warning: Tucker tensor not available or incomplete\n');
end

%% Best Configuration Detailed Analysis
fprintf('\n=== Generating Detailed Analysis for Best Configuration ===\n');

figure('Position', [100, 100, 1400, 800]);

% Plot 1: Convergence (objective + constraint)
subplot(2, 3, 1);
yyaxis left;
semilogy(1:info.iter, info.obj_values, 'b-', 'LineWidth', 1.5);
ylabel('Objective');
set(gca, 'YColor', 'b');
yyaxis right;
semilogy(1:info.iter, info.constraint_violations, 'r-', 'LineWidth', 1.5);
ylabel('Constraint Violation');
set(gca, 'YColor', 'r');
xlabel('Iteration');
title('Convergence History');
grid on;

% Plot 2: Error evolution
subplot(2, 3, 2);
if isfield(info, 'errors') && ~isempty(info.errors)
    semilogy(1:length(info.errors), info.errors, 'g-', 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('Reconstruction Error');
    title('Error vs Iteration');
    grid on;
else
    text(0.5, 0.5, 'No error tracking', 'HorizontalAlignment', 'center');
    axis off;
end

% Plot 3: Rank comparison
subplot(2, 3, 3);
bar([1, 2], [r, results.rank(best_idx)]);
set(gca, 'XTickLabel', {'True', 'Recovered'});
ylabel('Rank');
title('Rank Comparison');
grid on;
ylim([0, max(r, results.rank(best_idx))+1]);

% Plot 4: Time vs accuracy trade-off
subplot(2, 3, 4);
scatter(results.time, results.recon_error, 100, 1:n_configs, 'filled');
hold on;
plot(results.time(best_idx), results.recon_error(best_idx), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Reconstruction Error');
title('Efficiency Trade-off');
colorbar;
ylabel(colorbar, 'Config #');
grid on;
set(gca, 'YScale', 'log');
legend('All configs', 'Best', 'Location', 'best');

% Plot 5: Iteration count vs accuracy
subplot(2, 3, 5);
scatter(results.iterations, results.recon_error, 100, 1:n_configs, 'filled');
hold on;
plot(results.iterations(best_idx), results.recon_error(best_idx), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Iterations');
ylabel('Reconstruction Error');
title('Convergence vs Accuracy');
colorbar;
ylabel(colorbar, 'Config #');
grid on;
set(gca, 'YScale', 'log');

% Plot 6: Recovered ranks
subplot(2, 3, 6);
bar(1:n_configs, results.rank);
hold on;
yline(r, 'r--', 'LineWidth', 2);
set(gca, 'XTick', 1:n_configs, 'XTickLabel', results.config_label, 'XTickLabelRotation', 45);
ylabel('Rank');
title('Recovered Rank');
legend('Recovered', 'True rank', 'Location', 'best');
grid on;

sgtitle(sprintf('Best Configuration: %s', results.config_label{best_idx}));

%% Matrix Visualization
fprintf('\n=== Matrix Visualization ===\n');

figure('Position', [100, 100, 1200, 400]);

% Ground truth
subplot(1, 3, 1);
imagesc(Xstar);
colorbar;
axis square;
title(sprintf('Ground Truth (rank=%d)', rank(Xstar, 1e-6)));
xlabel('Column');
ylabel('Row');

% Recovered
subplot(1, 3, 2);
imagesc(X_recovered);
colorbar;
axis square;
title(sprintf('Recovered (rank=%d)', rank(X_recovered, 1e-6)));
xlabel('Column');
ylabel('Row');

% Difference
subplot(1, 3, 3);
imagesc(abs(X_recovered - Xstar));
colorbar;
axis square;
title(sprintf('Absolute Difference (err=%.2e)', recon_error));
xlabel('Column');
ylabel('Row');

sgtitle(sprintf('Matrix Comparison - %s', results.strategy_label));

%% Summary
fprintf('\n=== Test Summary ===\n');
fprintf('Test configuration:\n');
fprintf('  Dimension: %d × %d\n', d, d);
fprintf('  Ground truth rank: %d\n', rank(Xstar, 1e-6));
fprintf('  Measurements: %d (%.1f × d²)\n', m, m / d^2);
fprintf('  Strategies tested: %d\n', n_strategies);
fprintf('  Rho values per strategy: %d\n', n_rho);
fprintf('  Rho range: [%.4f, %.4f]\n', min(rho_values), max(rho_values));

fprintf('\nBest overall configuration:\n');
fprintf('  Strategy: %s\n', results.strategy_label);
fprintf('  Spectral init: %s, α: %.2f, Adaptive rho: %s\n', ...
        mat2str(results.use_spectral), results.over_relax, mat2str(results.adapt_rho));
fprintf('  Best initial rho: %.4f → final rho: %.4f\n', best_rho, results.final_rho(best_rho_idx));
fprintf('  Iterations: %d / %d\n', info.iter, max_iter);
fprintf('  Total time: %.4f seconds\n', info.total_time);
fprintf('  Time per iteration: %.4f seconds\n', mean(info.times));
fprintf('  Converged: %s\n', mat2str(info.converged));

fprintf('\nReconstruction quality:\n');
fprintf('  Reconstruction error: %.6e\n', recon_error);
fprintf('  Relative error: %.6e\n', rel_error);
fprintf('  Recovered rank: %d (true: %d)\n', rank(X_recovered, 1e-6), r);
fprintf('  Matrix symmetry error: %.6e\n', matrix_symmetry_error);

fprintf('\nConfiguration summary:\n');
for idx = 1:n_configs
    if results.converged(idx)
        converged_str = 'Yes';
    else
        converged_str = 'No';
    end
    fprintf('  %s: err=%.2e, iter=%d, time=%.3fs, conv=%s\n', ...
            results.config_label{idx}, results.recon_error(idx), results.iterations(idx), ...
            results.time(idx), converged_str);
end

fprintf('\nKey findings:\n');
[worst_error, worst_idx] = max(results.recon_error);
improvement = (worst_error - best_error) / worst_error * 100;
fprintf('  Best-to-worst error ratio: %.1f%% improvement\n', improvement);
fprintf('  Optimal PCG max iter: %d\n', results.pcg_maxit(best_idx));
fprintf('  PCG preconditioner beneficial: %s\n', mat2str(results.use_precond(best_idx)));

% Analyze preconditioner impact
% Compare pairs: configs with same settings except preconditioner
precond_pairs = [1, 2; 3, 4; 5, 6; 7, 8];
fprintf('\nPreconditioner impact analysis:\n');
for pair_idx = 1:size(precond_pairs, 1)
    no_precond_idx = precond_pairs(pair_idx, 1);
    with_precond_idx = precond_pairs(pair_idx, 2);
    
    err_no = results.recon_error(no_precond_idx);
    err_with = results.recon_error(with_precond_idx);
    time_no = results.time(no_precond_idx);
    time_with = results.time(with_precond_idx);
    
    err_ratio = err_no / err_with;
    time_ratio = time_no / time_with;
    
    fprintf('  PCG maxit=%d: Precond improves error by %.2fx, time ratio=%.2f\n', ...
            results.pcg_maxit(no_precond_idx), err_ratio, time_ratio);
end

% Success criterion
if best_error < 0.01 && results.converged(best_idx)
    fprintf('\n✓ Test PASSED: Excellent reconstruction with convergence\n');
elseif best_error < 0.1 && results.converged(best_idx)
    fprintf('\n✓ Test PASSED: Good reconstruction with convergence\n');
elseif best_error < 0.5
    fprintf('\n~ Test PARTIAL: Moderate reconstruction\n');
else
    fprintf('\n✗ Test FAILED: Poor reconstruction\n');
end

fprintf('\n=== Test Complete ===\n');
