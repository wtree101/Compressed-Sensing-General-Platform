%% Test Tensor Nuclear Norm Minimization Solver v2
% This script tests the solve_tensor_nuclear_norm_v2 function for phase retrieval
% using tensor nuclear norm minimization (convex relaxation) with the detailed
% ADMM formulation.
%
% Test objectives:
%   1. Run solver with default settings
%   2. Visualize objective and error convergence curves
%   3. Compare primal and measurement residuals
%
% ADMM formulation:
%   min_T  sum_{k=1}^4 lambda_k ||T_(k)||_*
%   s.t.   y_i = <A_i, T> / sqrt(m)
%
% with separate penalty parameters: rho_k (unfolding), rho_m (measurement)

clear; clc;

fprintf('=== Test Tensor Nuclear Norm Minimization v2 ===\n\n');

%% Test Parameters
d = 20;              % Dimension
r = 1;               % Rank of ground truth matrix
m = 400;             % Number of measurements

fprintf('Test Configuration:\n');
fprintf('  Dimension d: %d\n', d);
fprintf('  Rank r: %d\n', r);
fprintf('  Measurements m: %d\n', m);
fprintf('  Measurement ratio m/d²: %.2f\n\n', m / d^2);

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

% Create true tensor T_true = X_true ⊗ X_true
fprintf('Creating ground truth tensor...\n');
T_true = create_tensor_from_matrix(Xstar, d);
fprintf('  T_true: %dx%dx%dx%d tensor\n', d, d, d, d);
fprintf('  ||T_true||_F: %.6f\n\n', norm(T_true(:)));

%% Create Measurement Operator
fprintf('Creating measurement operator...\n');

% Random Gaussian measurements
A_cells = cell(m, 1);
for i = 1:m
    Ai = randn(d, d);
    A_cells{i} = (Ai + Ai') / 2;  % Symmetrize for consistency
end

% Create TuckerOperator for efficient operations
operator = TuckerOperator(A_cells, 'order', 4, 'symmetric', false, ...
                          'operator_type', 'kronecker');

fprintf('  Created %d measurement matrices\n', m);
fprintf('  Operator type: %s\n', operator.operator_type);
fprintf('  Tensor dimensions: [%s]\n\n', num2str(operator.dims));

%% Generate Measurements (phase retrieval)
fprintf('Generating measurements...\n');

% Compute y_i = |<A_i, X>|^2 / sqrt(m) for phase retrieval model
y = zeros(m, 1);
for i = 1:m
    y(i) = abs(trace(A_cells{i}' * Xstar))^2 / sqrt(m);
end

fprintf('  Measurements generated\n');
fprintf('  Measurement vector norm: %.6f\n\n', norm(y));

%% Solve using Tensor Nuclear Norm Minimization v2
fprintf('=== Running Tensor Nuclear Norm Solver v2 ===\n\n');

% Solver parameters
max_iter = 200;
lambda = [1,1,1,1];  % Equal weights for all modes
rho_k = 1000;            % Penalty for unfolding constraints
rho_m = 10000;            % Penalty for measurement constraints
verbose = 1;

fprintf('Solver parameters:\n');
fprintf('  max_iter: %d\n', max_iter);
fprintf('  lambda: [%.1f, %.1f, %.1f, %.1f]\n', lambda);
fprintf('  rho_k (unfolding): %.2f\n', rho_k);
fprintf('  rho_m (measurement): %.2f\n', rho_m);
fprintf('  pcg_tol: 1e-3 (default)\n');
fprintf('  pcg_maxit: 100 (default)\n\n');

% Call solver v2
fprintf('Running solver v2...\n\n');
tic;
[X_recovered, info] = solve_tensor_nuclear_norm_v2(operator, y, d, ...
    'max_iter', max_iter, ...
    'lambda', lambda, ...
    'rho_k', rho_k, ...
    'rho_m', rho_m, ...
    'verbose', verbose, ...
    'X_true', Xstar);
elapsed_time = toc;

% Compute reconstruction error
[recon_error, X_recovered_aligned] = rectify_sign_ambiguity(X_recovered, Xstar);
rel_error = norm(X_recovered_aligned - Xstar, 'fro') / norm(Xstar, 'fro');

%% Results Summary
fprintf('\n=== Results Summary ===\n\n');

fprintf('Convergence:\n');
fprintf('  Iterations: %d\n', info.iter);
fprintf('  Converged: %s\n', mat2str(info.converged));
fprintf('  Total time: %.4f seconds\n', elapsed_time);
fprintf('  Time per iteration: %.4f seconds\n\n', elapsed_time / info.iter);

fprintf('Final values:\n');
fprintf('  Objective: %.6e\n', info.obj_values(end));
fprintf('  Primal residual: %.6e\n', info.primal_residuals(end));
fprintf('  Measurement residual: %.6e\n', info.measurement_residuals(end));
fprintf('  Reconstruction error: %.6e\n', recon_error);
fprintf('  Relative error: %.6e\n\n', rel_error);

fprintf('Matrix properties:\n');
fprintf('  Recovered rank: %d\n', rank(X_recovered, 1e-6));
fprintf('  Ground truth rank: %d\n', rank(Xstar, 1e-6));
fprintf('  Recovered norm: %.6f\n', norm(X_recovered, 'fro'));
fprintf('  Ground truth norm: %.6f\n\n', norm(Xstar, 'fro'));

%% Visualization: Convergence Curves
fprintf('Generating convergence plots...\n\n');

figure('Position', [100, 100, 1400, 900], 'Name', 'ADMM v2 Convergence Analysis');

% Plot 1: Objective function
subplot(2, 3, 1);
semilogy(1:info.iter, info.obj_values, 'b-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Objective Value');
title('Objective Function');
grid on;

% Plot 2: Primal residual (unfolding constraints)
subplot(2, 3, 2);
semilogy(1:info.iter, info.primal_residuals, 'r-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Primal Residual ||W_k - T_{(k)}||');
title('Primal Residual (Unfolding Constraints)');
grid on;

% Plot 3: Measurement residual
subplot(2, 3, 3);
semilogy(1:info.iter, info.measurement_residuals, 'g-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Measurement Residual');
title('Measurement Constraint Violation');
grid on;

% Plot 4: Reconstruction error
subplot(2, 3, 4);
if isfield(info, 'errors') && ~isempty(info.errors)
    semilogy(1:info.iter, info.errors, 'm-', 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('Reconstruction Error');
    title('Error vs Ground Truth');
    grid on;
else
    text(0.5, 0.5, 'No error tracking', 'HorizontalAlignment', 'center');
    axis off;
end

% Plot 5: Combined residuals (dual axis)
subplot(2, 3, 5);
yyaxis left;
semilogy(1:info.iter, info.primal_residuals, 'r-', 'LineWidth', 1.5);
ylabel('Primal Residual');
set(gca, 'YColor', 'r');
yyaxis right;
semilogy(1:info.iter, info.measurement_residuals, 'g-', 'LineWidth', 1.5);
ylabel('Measurement Residual');
set(gca, 'YColor', 'g');
xlabel('Iteration');
title('Combined Residuals');
grid on;

% Plot 6: Time per iteration
subplot(2, 3, 6);
plot(1:info.iter, info.times, 'k-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Time (seconds)');
title('Computation Time per Iteration');
grid on;

sgtitle(sprintf('ADMM v2: d=%d, m=%d, rho_k=%.2f, rho_m=%.2f', d, m, rho_k, rho_m));

%% Visualization: Matrix Comparison
fprintf('Generating matrix comparison plots...\n\n');

figure('Position', [150, 150, 1200, 400], 'Name', 'Matrix Recovery Results');

% Ground truth
subplot(1, 3, 1);
imagesc(Xstar);
colorbar;
axis square;
title(sprintf('Ground Truth (rank %d)', rank(Xstar, 1e-6)));
xlabel('Column');
ylabel('Row');

% Recovered matrix (aligned)
subplot(1, 3, 2);
imagesc(X_recovered_aligned);
colorbar;
axis square;
title(sprintf('Recovered (rank %d)', rank(X_recovered_aligned, 1e-6)));
xlabel('Column');
ylabel('Row');

% Difference
subplot(1, 3, 3);
imagesc(abs(X_recovered_aligned - Xstar));
colorbar;
axis square;
title(sprintf('Absolute Difference (error=%.2e)', recon_error));
xlabel('Column');
ylabel('Row');

sgtitle('Matrix Recovery Comparison');

%% Tensor Error Analysis
fprintf('=== Tensor Error Analysis ===\n\n');

% Extract recovered tensor
T_recovered = info.T;

% Compute comprehensive tensor errors
fprintf('Tensor-level errors:\n');

% 1. Absolute Frobenius error
tensor_error_abs = norm(T_recovered(:) - T_true(:));
fprintf('  ||T_recovered - T_true||_F = %.6e\n', tensor_error_abs);

% 2. Relative Frobenius error
tensor_error_rel = tensor_error_abs / norm(T_true(:));
fprintf('  Relative error: %.6e\n', tensor_error_rel);

% 3. Elementwise errors
tensor_error_max = max(abs(T_recovered(:) - T_true(:)));
tensor_error_mean = mean(abs(T_recovered(:) - T_true(:)));
fprintf('  Max absolute element error: %.6e\n', tensor_error_max);
fprintf('  Mean absolute element error: %.6e\n\n', tensor_error_mean);

% Compute mode unfolding errors
fprintf('Mode unfolding errors:\n');
unfolding_norms = zeros(1, 4);
unfolding_norms_true = zeros(1, 4);
unfolding_ranks = zeros(1, 4);
unfolding_ranks_true = zeros(1, 4);
unfolding_nuclear_norms = zeros(1, 4);
unfolding_nuclear_norms_true = zeros(1, 4);
unfolding_errors = zeros(1, 4);
unfolding_errors_rel = zeros(1, 4);

for k = 1:4
    % Recovered tensor unfolding
    T_k = tensor_mode_unfold_local(T_recovered, k);
    unfolding_norms(k) = norm(T_k, 'fro');
    unfolding_ranks(k) = rank(T_k, 1e-6);
    unfolding_nuclear_norms(k) = sum(svd(T_k));
    
    % True tensor unfolding
    T_true_k = tensor_mode_unfold_local(T_true, k);
    unfolding_norms_true(k) = norm(T_true_k, 'fro');
    unfolding_ranks_true(k) = rank(T_true_k, 1e-6);
    unfolding_nuclear_norms_true(k) = sum(svd(T_true_k));
    
    % Compute errors
    unfolding_errors(k) = norm(T_k - T_true_k, 'fro');
    unfolding_errors_rel(k) = unfolding_errors(k) / unfolding_norms_true(k);
    
    fprintf('  Mode %d:\n', k);
    fprintf('    Recovered: ||T_(%d)||_F = %.6f, rank = %d, ||T_(%d)||_* = %.6f\n', ...
        k, unfolding_norms(k), unfolding_ranks(k), k, unfolding_nuclear_norms(k));
    fprintf('    True:      ||T_(%d)||_F = %.6f, rank = %d, ||T_(%d)||_* = %.6f\n', ...
        k, unfolding_norms_true(k), unfolding_ranks_true(k), k, unfolding_nuclear_norms_true(k));
    fprintf('    Error:     ||T_(%d) - T_true_(%d)||_F = %.6e (relative: %.6e)\n', ...
        k, k, unfolding_errors(k), unfolding_errors_rel(k));
    fprintf('    Norm diff: %.6e, Rank diff: %d, Nuclear norm diff: %.6e\n\n', ...
        abs(unfolding_norms(k) - unfolding_norms_true(k)), ...
        abs(unfolding_ranks(k) - unfolding_ranks_true(k)), ...
        abs(unfolding_nuclear_norms(k) - unfolding_nuclear_norms_true(k)));
end

% Summary of unfolding errors
fprintf('Unfolding error summary:\n');
fprintf('  Average unfolding error: %.6e\n', mean(unfolding_errors));
fprintf('  Max unfolding error: %.6e\n', max(unfolding_errors));
fprintf('  Min unfolding error: %.6e\n\n', min(unfolding_errors));

% Check symmetry of recovered tensor unfoldings
fprintf('Symmetry analysis of recovered tensor unfoldings:\n');
symmetry_errors = zeros(6, 1);
pair_idx = 1;
for k1 = 1:3
    for k2 = (k1+1):4
        T_k1 = tensor_mode_unfold_local(T_recovered, k1);
        T_k2 = tensor_mode_unfold_local(T_recovered, k2);
        
        % For 4th-order tensor T = X ⊗ X, all mode unfoldings should have similar structure
        % Compare Frobenius norms as a measure of similarity
        symmetry_errors(pair_idx) = abs(norm(T_k1, 'fro') - norm(T_k2, 'fro'));
        fprintf('  ||T_(%d)||_F vs ||T_(%d)||_F: diff = %.6e\n', ...
            k1, k2, symmetry_errors(pair_idx));
        pair_idx = pair_idx + 1;
    end
end

avg_symmetry_error = mean(symmetry_errors);
max_symmetry_error = max(symmetry_errors);
fprintf('\n  Average norm difference: %.6e\n', avg_symmetry_error);
fprintf('  Maximum norm difference: %.6e\n', max_symmetry_error);

% Check if norms are approximately equal (indicating symmetry)
symmetry_tolerance = 1e-2;
is_symmetric = max_symmetry_error < symmetry_tolerance;
if is_symmetric
    fprintf('  ✓ Tensor unfoldings are symmetric (similar norms)\n');
else
    fprintf('  ✗ Tensor unfoldings show asymmetry\n');
end

%% Visualization: Tensor Unfolding Structure
fprintf('\nGenerating tensor unfolding visualization...\n\n');

figure('Position', [200, 200, 1400, 800], 'Name', 'Tensor Unfolding Analysis');

for k = 1:4
    subplot(2, 4, k);
    T_k = tensor_mode_unfold_local(T_recovered, k);
    
    % Show singular value spectrum
    s_k = svd(T_k);
    semilogy(s_k, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    xlabel('Index');
    ylabel('Singular Value');
    title(sprintf('Mode %d: Singular Values', k));
    grid on;
    
    subplot(2, 4, k+4);
    % Show first few singular vectors (heatmap)
    [U_k, ~, ~] = svd(T_k, 'econ');
    num_show = min(5, size(U_k, 2));
    imagesc(U_k(:, 1:num_show));
    colorbar;
    xlabel('Singular Vector Index');
    ylabel('Element');
    title(sprintf('Mode %d: First %d Left Singular Vectors', k, num_show));
end

sgtitle('Tensor Unfolding Structure Analysis');

%% Comprehensive Error Summary Table
fprintf('=== Comprehensive Error Summary ===\n\n');

fprintf('┌─────────────────────────────────────────────────────────────┐\n');
fprintf('│                    ERROR COMPARISON TABLE                   │\n');
fprintf('├─────────────────────────────────────────────────────────────┤\n');

% Matrix-level errors
fprintf('│ Matrix Level (X)                                            │\n');
fprintf('│   Reconstruction error:        %.6e                     │\n', recon_error);
fprintf('│   Relative error:              %.6e                     │\n', rel_error);
fprintf('│   Rank (recovered/true):       %d / %d                     │\n', ...
    rank(X_recovered, 1e-6), rank(Xstar, 1e-6));
fprintf('├─────────────────────────────────────────────────────────────┤\n');

% Tensor-level errors
fprintf('│ Tensor Level (T = X ⊗ X)                                    │\n');
fprintf('│   Absolute error ||T-T_true||: %.6e                     │\n', tensor_error_abs);
fprintf('│   Relative error:              %.6e                     │\n', tensor_error_rel);
fprintf('│   Max element error:           %.6e                     │\n', tensor_error_max);
fprintf('│   Mean element error:          %.6e                     │\n', tensor_error_mean);
fprintf('│   Norm (recovered/true):       %.6f / %.6f              │\n', ...
    norm(T_recovered(:)), norm(T_true(:)));
fprintf('├─────────────────────────────────────────────────────────────┤\n');

% Unfolding-level errors
fprintf('│ Mode Unfolding Level                                        │\n');
for k = 1:4
    fprintf('│   Mode %d error:                %.6e                     │\n', k, unfolding_errors(k));
    fprintf('│   Mode %d relative error:       %.6e                     │\n', k, unfolding_errors_rel(k));
end
fprintf('│   Average unfolding error:     %.6e                     │\n', mean(unfolding_errors));
fprintf('│   Max unfolding error:         %.6e                     │\n', max(unfolding_errors));
fprintf('├─────────────────────────────────────────────────────────────┤\n');

% Convergence metrics
fprintf('│ Convergence Metrics                                         │\n');
fprintf('│   Iterations:                  %d / %d                     │\n', info.iter, max_iter);
fprintf('│   Converged:                   %s                           │\n', mat2str(info.converged));
fprintf('│   Final objective:             %.6e                     │\n', info.obj_values(end));
fprintf('│   Final primal residual:       %.6e                     │\n', info.primal_residuals(end));
fprintf('│   Final measurement residual:  %.6e                     │\n', info.measurement_residuals(end));
fprintf('│   Total time:                  %.4f seconds               │\n', elapsed_time);
fprintf('└─────────────────────────────────────────────────────────────┘\n\n');

% Error relationship verification
fprintf('Error relationship verification:\n');
fprintf('  Matrix error (X):        %.6e\n', recon_error);
fprintf('  Tensor error (T=X⊗X):    %.6e\n', tensor_error_rel);
fprintf('  Expected relationship:   tensor_error ≈ 2 * matrix_error (approximately)\n');
fprintf('  Actual ratio:            %.4f\n\n', tensor_error_rel / recon_error);

%% Final Assessment
fprintf('=== Final Assessment ===\n\n');

if recon_error < 0.01 && info.converged
    fprintf('✓ Test PASSED: Excellent reconstruction with convergence\n');
elseif recon_error < 0.1 && info.converged
    fprintf('✓ Test PASSED: Good reconstruction with convergence\n');
elseif recon_error < 0.5
    fprintf('~ Test PARTIAL: Moderate reconstruction\n');
else
    fprintf('✗ Test FAILED: Poor reconstruction\n');
end

fprintf('\n=== Test Complete ===\n');

%% Helper function for unfolding
function T_mat = tensor_mode_unfold_local(T, mode)
    % Mode-k unfolding (matricization) of tensor T
    sz = size(T);
    n_modes = length(sz);
    
    % Permute so mode is first
    perm = [mode, 1:mode-1, mode+1:n_modes];
    T_perm = permute(T, perm);
    
    % Reshape to matrix
    T_mat = reshape(T_perm, sz(mode), []);
end
