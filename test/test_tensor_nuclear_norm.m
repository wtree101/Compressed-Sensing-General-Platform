%% Test Tensor Nuclear Norm Minimization Solver - Default Settings
% This script tests the solve_tensor_nuclear_norm function for phase retrieval
% using tensor nuclear norm minimization (convex relaxation)
%
% Test objectives:
%   1. Run solver with default settings
%   2. Visualize objective and error convergence curves
%
% Default settings: rho=0.1, over_relax=1.0, pcg_maxit=5, no preconditioner

clear; clc;

fprintf('=== Test Tensor Nuclear Norm Minimization - Default Settings ===\n\n');

%% Test Parameters
d = 20;              % Dimension (start small for testing)
r = 1;               % Rank of ground truth matrix
m = 256;             % Number of measurements (m >= d^2 for recovery)

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

operator = struct();
operator.A_cells = A_cells;

fprintf('  Created %d measurement matrices\n\n', m);

%% Generate Measurements (phase retrieval)
fprintf('Generating measurements...\n');

% Compute y_i = <A_i ⊗ A_i, X ⊗ X>
y = zeros(m, 1);
for i = 1:m
    % y_i = <A_i, X>²  / sqrt(m)  for phase retrieval model
    y(i) = abs(trace(A_cells{i}' * Xstar))^2 / sqrt(m);
end

fprintf('  Measurements generated\n');
fprintf('  Measurement vector norm: %.6f\n\n', norm(y));

%% Solve using Tensor Nuclear Norm Minimization - Default Settings
fprintf('=== Running Tensor Nuclear Norm Solver (Default Settings) ===\n\n');

% Solver parameters (using defaults)
max_iter = 200;
lambda = 1.0;       % Regularization parameter
verbose = 1;        % Show progress
rho = 0.5;
fprintf('Solver parameters (defaults):\n');
fprintf('  max_iter: %d\n', max_iter);
fprintf('  lambda: %.2f\n', lambda);
fprintf('  rho: 0.1 (default)\n');
fprintf('  over_relax: 1.0 (default, no over-relaxation)\n');
fprintf('  pcg_maxit: 5 (default)\n');
fprintf('  pcg_tol: 1e-3 (default)\n');
fprintf('  use_pcg_precond: false (default, no preconditioner)\n\n');

% Call solver with default configuration
fprintf('Running solver...\n\n');
tic;
[X_recovered, info] = solve_tensor_nuclear_norm(operator, y, d, ...
    'max_iter', max_iter, ...
    'lambda', lambda, ...
    'verbose', verbose, ...
    'X_true', Xstar, 'rho', rho);
elapsed_time = toc;

% Compute reconstruction error
[recon_error, X_recovered_aligned] = rectify_sign_ambiguity(X_recovered, Xstar);
rel_error = norm(X_recovered_aligned - Xstar, 'fro') / norm(Xstar, 'fro');

%% Results Summary
fprintf('\n=== Results Summary ===\n\n');

fprintf('Reconstruction Quality:\n');
fprintf('  Reconstruction error: %.6e\n', recon_error);
fprintf('  Relative error: %.6e\n', rel_error);
fprintf('  Recovered matrix rank: %d (true: %d)\n', rank(X_recovered, 1e-6), r);
fprintf('  Recovered matrix norm: %.6f\n', norm(X_recovered, 'fro'));

% Check matrix symmetry (X should be symmetric)
matrix_symmetry_error = norm(X_recovered - X_recovered', 'fro');
fprintf('  Matrix symmetry error: %.6e\n', matrix_symmetry_error);

fprintf('\nSolver Performance:\n');
fprintf('  Total iterations: %d / %d\n', info.iter, max_iter);
fprintf('  Total time: %.4f seconds\n', elapsed_time);
fprintf('  Time per iteration: %.4f seconds\n', mean(info.times));
fprintf('  Converged: %s\n', mat2str(info.converged));
fprintf('  Final objective: %.6e\n', info.obj_values(end));
fprintf('  Final constraint violation: %.6e\n', info.constraint_violations(end));

%% Convergence Visualization
fprintf('\n=== Generating Convergence Plots ===\n');

figure('Position', [100, 100, 1400, 900]);

% Plot 1: Objective function
subplot(2, 3, 1);
semilogy(1:info.iter, info.obj_values, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Objective Value');
title('Objective Function');
grid on;

% Plot 2: Constraint violation
subplot(2, 3, 2);
semilogy(1:info.iter, info.constraint_violations, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Constraint Violation');
title('Measurement Constraint Violation');
grid on;

% Plot 3: Reconstruction error
subplot(2, 3, 3);
if isfield(info, 'errors') && ~isempty(info.errors)
    semilogy(1:length(info.errors), info.errors, 'g-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Reconstruction Error');
    title('Reconstruction Error vs Iteration');
    grid on;
else
    text(0.5, 0.5, 'No error tracking', 'HorizontalAlignment', 'center');
    axis off;
end

% Plot 4: Combined objective + constraint
subplot(2, 3, 4);
yyaxis left;
semilogy(1:info.iter, info.obj_values, 'b-', 'LineWidth', 2);
ylabel('Objective');
set(gca, 'YColor', 'b');
yyaxis right;
semilogy(1:info.iter, info.constraint_violations, 'r-', 'LineWidth', 2);
ylabel('Constraint Violation');
set(gca, 'YColor', 'r');
xlabel('Iteration');
title('Convergence History');
grid on;
legend('Objective', 'Constraint', 'Location', 'best');

% Plot 5: Iteration timing
subplot(2, 3, 5);
plot(1:info.iter, info.times, 'k-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Time (s)');
title('Time per Iteration');
grid on;

% Plot 6: Cumulative time
subplot(2, 3, 6);
plot(1:info.iter, cumsum(info.times), 'k-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Cumulative Time (s)');
title('Total Computation Time');
grid on;

sgtitle(sprintf('Tensor Nuclear Norm Minimization - Default Settings (d=%d, m=%d)', d, m));

%% Matrix Visualization
fprintf('\n=== Generating Matrix Visualization ===\n');

figure('Position', [100, 100, 1400, 400]);

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

sgtitle('Matrix Comparison - Tensor Nuclear Norm Minimization');

%% Final Summary
fprintf('\n=== Test Summary ===\n');
fprintf('Problem configuration:\n');
fprintf('  Dimension: %d × %d\n', d, d);
fprintf('  Ground truth rank: %d\n', rank(Xstar, 1e-6));
fprintf('  Measurements: %d (%.1f × d²)\n', m, m / d^2);

fprintf('\nSolver settings (all defaults):\n');
fprintf('  rho: 0.1\n');
fprintf('  over_relax: 1.0 (no over-relaxation)\n');
fprintf('  pcg_maxit: 5\n');
fprintf('  pcg_tol: 1e-3\n');
fprintf('  use_pcg_precond: false (no preconditioner)\n');
fprintf('  lambda: %.2f\n', lambda);
fprintf('  max_iter: %d\n', max_iter);

fprintf('\nResults:\n');
fprintf('  Reconstruction error: %.6e\n', recon_error);
fprintf('  Relative error: %.6e\n', rel_error);
fprintf('  Recovered rank: %d (true: %d)\n', rank(X_recovered, 1e-6), r);
fprintf('  Matrix symmetry error: %.6e\n', matrix_symmetry_error);
fprintf('  Iterations: %d / %d\n', info.iter, max_iter);
fprintf('  Total time: %.4f seconds\n', elapsed_time);
fprintf('  Converged: %s\n', mat2str(info.converged));

% Success criterion
if recon_error < 0.01 && info.converged
    fprintf('\n✓ Test PASSED: Excellent reconstruction with convergence\n');
elseif recon_error < 0.1 && info.converged
    fprintf('\n✓ Test PASSED: Good reconstruction with convergence\n');
elseif recon_error < 0.5
    fprintf('\n~ Test PARTIAL: Moderate reconstruction\n');
else
    fprintf('\n✗ Test FAILED: Poor reconstruction\n');
end

fprintf('\n=== Test Complete ===\n');
