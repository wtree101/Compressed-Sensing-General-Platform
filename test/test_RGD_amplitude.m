%% Test Script: Comparison of Local Refinement Methods
% This script compares different local refinement methods for low-rank matrix recovery:
%   1. RGD (Riemannian Gradient Descent, no preconditioner)
%   2. PRGD (Preconditioned Riemannian Gradient Descent)
%   3. PGD (Projected Gradient Descent, standard)
%
% Structure: Fixed Initialization + Different Local Refinement
%   - Initialization: Projected Power Method (same for all methods)
%   - Local Refinement: Compare RGD, PRGD, PGD

clear; clc;
addpath(genpath('.'));

fprintf('=== Comparison of Local Refinement Methods ===\n');
fprintf('Structure: Fixed Initialization + Different Local Refinement\n\n');

%% Problem Setup
d = 20;              % Matrix dimension
r = 1;               % Target rank
m = 250;             % Number of measurements
kappa = 2;           % Condition number

fprintf('Problem configuration:\n');
fprintf('  Matrix size: %dx%d\n', d, d);
fprintf('  Rank: %d\n', r);
fprintf('  Measurements: %d\n', m);
fprintf('  Measurement ratio: %.2f\n\n', m / d^2);

%% Generate Ground Truth
U_true = randn(d, r);
Xstar = abs(U_true) * abs(U_true)';  % Symmetric rank-r matrix
Xstar = Xstar / norm(Xstar, 'fro');

fprintf('Ground truth properties:\n');
fprintf('  ||X*||_F = %.4f\n', norm(Xstar, 'fro'));
fprintf('  rank(X*) = %d\n', rank(Xstar, 1e-6));
fprintf('  Spectrum range: [%.4e, %.4e]\n\n', min(eig(Xstar)), max(eig(Xstar)));

%% Nonlinear Function Configuration
% Nonlinear function for measurements (following Phasediagram structure)
nonlinear_func = @(y) abs(y);  % Phase retrieval model (amplitude-only measurements)
% Set to [] for linear measurements, @(y) abs(y) for amplitude measurements

%% Generate Measurements (following onetrial_Mat structure)
% Generate measurement matrix A (m x d^2)
% Each row corresponds to one measurement
problem_flag = 0;  % 0 for sensing, 1 for completion
A = generate_A(problem_flag, m, d, d, struct());

% Create operator structure (following onetrial_Mat.m)
operator.A = @(X) A * X(:);  % Forward operator: matrix to measurements
operator.A_star = @(y_vec) reshape(A' * y_vec, [d, d]);  % Adjoint operator: measurements to matrix

% Generate measurements with optional nonlinearity
y_true = operator.A(Xstar) / sqrt(m);
if ~isempty(nonlinear_func)
    y = nonlinear_func(y_true);  % Apply nonlinear function (e.g., amplitude)
    fprintf('Measurement type: Amplitude-only (Phase Retrieval)\n');
else
    y = y_true;  % Linear measurements
    fprintf('Measurement type: Linear\n');
end

fprintf('Measurement properties:\n');
fprintf('  ||y||_2 = %.4f\n', norm(y));
fprintf('  y range: [%.4e, %.4e]\n', min(y), max(y));
fprintf('  Operator: %dx%d matrix A for measurements\n\n', size(A, 1), size(A, 2));

%% Initialization (Same for all methods)
fprintf('=== Phase 1: Initialization ===\n');
fprintf('Method: Projected Power Method\n');

% Available initialization methods in Initialization_groundtruth/:
%   - initialize_power_method.m: Spectral method via power iterations (good for general problems)
%   - initialize_tensor_lift.m: Tensor lifting method (for tensor recovery)
%   - initialize_tensor_lift_efficient.m: Memory-efficient tensor lifting
%   - initialize_tensor_lift_tucker_spectral.m: Tucker decomposition based tensor lifting
%   - initialize_tensor_nuclear_norm.m: Tensor nuclear norm minimization (v1)
%   - initialize_tensor_nuclear_norm_v2.m: Improved tensor nuclear norm minimization (v2)
%   - init_matrix.m: Basic matrix initialization utilities
%   - init_random_vector.m: Random initialization for vectors
%   - init_zero_vector.m: Zero initialization for vectors
%   - init_matching_pursuit_vector.m: Matching pursuit for sparse vectors
%   - Initialization_random.m: Random initialization strategy
%   - Initialization.m: General initialization framework

T_power = 20;  % Number of power iterations
init_params = struct();
init_params.T_power = T_power;
init_params.r = r;
init_params.projection = @(X) rank_projection(X, r);
init_params.prefunc = @(y) y;  % No preprocessing for amplitude
init_params.verbose = 0;

% Run initialization
tic;
[X0, ~, init_history] = initialize_power_method(y, operator, d, d, init_params);
init_time = toc;

% Compute initial error
[init_error, ~] = rectify_sign_ambiguity(X0, Xstar);

fprintf('Initialization complete:\n');
fprintf('  Time: %.4f seconds\n', init_time);
fprintf('  Initial error: %.6e\n', init_error);
fprintf('  Rank: %d\n\n', rank(X0, 1e-6));

%% Phase 2: Local Refinement - Compare Different Methods
fprintf('=== Phase 2: Local Refinement (starting from same initialization) ===\n\n');

T_refine = 200;  % Number of refinement iterations

%% Method 1: PRGD (Preconditioned Riemannian Gradient Descent)
fprintf('--- Method 1: PRGD (Preconditioned RGD, Factorized) ---\n');
mu_prgd = 0.1;   % Step size for PRGD
fprintf('Using stepsize: μ = %.2f\n', mu_prgd);
params_prgd = struct();
params_prgd.T = T_refine;
params_prgd.mu = mu_prgd;
params_prgd.r = r;
params_prgd.use_preconditioner = true;
params_prgd.epsilon = 1e-8;
params_prgd.Xstar = Xstar;
params_prgd.use_spectral_init = false;  % Start from provided X0
params_prgd.verbose = 0;
params_prgd.return_factorized = false;

tic;
[output_prgd, X_prgd] = solve_RGD_amplitude(X0, [], y, operator, d, d, [], m, params_prgd);
time_prgd = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_prgd, time_prgd + init_time);
fprintf('  Initial error: %.6e\n', output_prgd.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_prgd.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_prgd.Error_Stand(1) / output_prgd.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_prgd.Error_function(end));
fprintf('  Rank: %d (factorized: %dx%d + %dx%d)\n\n', ...
        rank(X_prgd, 1e-6), size(output_prgd.U, 1), size(output_prgd.U, 2), ...
        size(output_prgd.V, 1), size(output_prgd.V, 2));

%% Method 2: RGD (Riemannian GD without Preconditioner)
fprintf('--- Method 2: RGD (Standard RGD, Factorized) ---\n');
mu_rgd = 1.0;    % Step size for RGD (larger, no preconditioner)
fprintf('Using stepsize: μ = %.2f\n', mu_rgd);
params_rgd = struct();
params_rgd.T = T_refine;
params_rgd.mu = mu_rgd;
params_rgd.r = r;
params_rgd.use_preconditioner = false;
params_rgd.Xstar = Xstar;
params_rgd.use_spectral_init = false;  % Start from provided X0
params_rgd.verbose = 0;
params_rgd.return_factorized = false;

tic;
[output_rgd, X_rgd] = solve_RGD_amplitude(X0, [], y, operator, d, d, [], m, params_rgd);
time_rgd = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_rgd, time_rgd + init_time);
fprintf('  Initial error: %.6e\n', output_rgd.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_rgd.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_rgd.Error_Stand(1) / output_rgd.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_rgd.Error_function(end));
fprintf('  Rank: %d (factorized: %dx%d + %dx%d)\n\n', ...
        rank(X_rgd, 1e-6), size(output_rgd.U, 1), size(output_rgd.U, 2), ...
        size(output_rgd.V, 1), size(output_rgd.V, 2));

%% Method 3: PGD (Standard Projected Gradient Descent)
fprintf('--- Method 3: PGD (Standard, for comparison) ---\n');
mu_pgd = 1.0;    % Step size for PGD (larger, no preconditioner)
fprintf('Using stepsize: μ = %.2f\n', mu_pgd);
params_pgd = struct();
params_pgd.T = T_refine;
params_pgd.mu = mu_pgd;
params_pgd.projection = @(X) rank_projection(X, r);
params_pgd.Xstar = Xstar;

tic;
[output_pgd, X_pgd] = solve_PGD_amplitude(X0, [], y, operator, d, d, [], m, params_pgd);
time_pgd = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_pgd, time_pgd + init_time);
fprintf('  Initial error: %.6e\n', output_pgd.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_pgd.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_pgd.Error_Stand(1) / output_pgd.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_pgd.Error_function(end));
fprintf('  Rank: %d\n\n', rank(X_pgd, 1e-6));

%% Method 4: Preconditioned PGD (PGD with diagonal preconditioner)
fprintf('--- Method 4: Preconditioned PGD (PPGD) ---\n');
mu_ppgd = 0.1;    % Step size for Preconditioned PGD (similar to PRGD)
fprintf('Using stepsize: μ = %.2f\n', mu_ppgd);
params_ppgd = struct();
params_ppgd.T = T_refine;
params_ppgd.mu = mu_ppgd;
params_ppgd.projection = @(X) rank_projection(X, r);
params_ppgd.use_preconditioner = true;  % Enable preconditioner
params_ppgd.epsilon = 1e-8;
params_ppgd.Xstar = Xstar;

tic;
[output_ppgd, X_ppgd] = solve_PGD_amplitude(X0, [], y, operator, d, d, [], m, params_ppgd);
time_ppgd = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_ppgd, time_ppgd + init_time);
fprintf('  Initial error: %.6e\n', output_ppgd.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_ppgd.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_ppgd.Error_Stand(1) / output_ppgd.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_ppgd.Error_function(end));
fprintf('  Rank: %d\n\n', rank(X_ppgd, 1e-6));

%% Method 5: Adaptive PRGD (Preconditioned RGD with Adaptive Stepsize)
fprintf('--- Method 5: Adaptive PRGD (with Line Search) ---\n');
mu_adaptive = 0.3;  % Initial stepsize for adaptive (same as PRGD)
fprintf('Using initial stepsize: μ = %.2f (will adapt via line search)\n', mu_adaptive);
params_adaptive = struct();
params_adaptive.T = T_refine;
params_adaptive.mu = mu_adaptive;  % Initial/default stepsize
params_adaptive.r = r;
params_adaptive.use_preconditioner = true;
params_adaptive.epsilon = 1e-8;
params_adaptive.Xstar = Xstar;
params_adaptive.use_spectral_init = false;  % Start from provided X0
params_adaptive.verbose = 0;
params_adaptive.return_factorized = false;
% Enable adaptive stepsize
params_adaptive.use_adaptive_stepsize = true;
params_adaptive.line_search_max_iter = 20;
params_adaptive.line_search_beta = 0.5;
params_adaptive.line_search_c = 1e-4;

tic;
[output_adaptive, X_adaptive] = solve_RGD_amplitude(X0, [], y, operator, d, d, [], m, params_adaptive);
time_adaptive = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_adaptive, time_adaptive + init_time);
fprintf('  Initial error: %.6e\n', output_adaptive.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_adaptive.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_adaptive.Error_Stand(1) / output_adaptive.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_adaptive.Error_function(end));
fprintf('  Rank: %d (factorized: %dx%d + %dx%d)\n', ...
        rank(X_adaptive, 1e-6), size(output_adaptive.U, 1), size(output_adaptive.U, 2), ...
        size(output_adaptive.V, 1), size(output_adaptive.V, 2));
fprintf('  Stepsize range: [%.4e, %.4e] (mean: %.4e)\n\n', ...
        min(output_adaptive.stepsize_history), max(output_adaptive.stepsize_history), ...
        mean(output_adaptive.stepsize_history));

%% Method 6: Adaptive RGD (RGD with Adaptive Stepsize, no preconditioner)
fprintf('--- Method 6: Adaptive RGD (with Line Search, no preconditioner) ---\n');
mu_adaptive_rgd = 0.5;  % Initial stepsize for adaptive RGD (same as fixed RGD)
fprintf('Using initial stepsize: μ = %.2f (will adapt via line search)\n', mu_adaptive_rgd);
params_adaptive_rgd = struct();
params_adaptive_rgd.T = T_refine;
params_adaptive_rgd.mu = mu_adaptive_rgd;  % Initial/default stepsize
params_adaptive_rgd.r = r;
params_adaptive_rgd.use_preconditioner = false;  % NO preconditioner (key difference from Method 4)
params_adaptive_rgd.Xstar = Xstar;
params_adaptive_rgd.use_spectral_init = false;  % Start from provided X0
params_adaptive_rgd.verbose = 0;
params_adaptive_rgd.return_factorized = false;
% Enable adaptive stepsize
params_adaptive_rgd.use_adaptive_stepsize = true;
params_adaptive_rgd.line_search_max_iter = 20;
params_adaptive_rgd.line_search_beta = 0.5;
params_adaptive_rgd.line_search_c = 1e-4;

tic;
[output_adaptive_rgd, X_adaptive_rgd] = solve_RGD_amplitude(X0, [], y, operator, d, d, [], m, params_adaptive_rgd);
time_adaptive_rgd = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_adaptive_rgd, time_adaptive_rgd + init_time);
fprintf('  Initial error: %.6e\n', output_adaptive_rgd.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_adaptive_rgd.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_adaptive_rgd.Error_Stand(1) / output_adaptive_rgd.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_adaptive_rgd.Error_function(end));
fprintf('  Rank: %d (factorized: %dx%d + %dx%d)\n', ...
        rank(X_adaptive_rgd, 1e-6), size(output_adaptive_rgd.U, 1), size(output_adaptive_rgd.U, 2), ...
        size(output_adaptive_rgd.V, 1), size(output_adaptive_rgd.V, 2));
fprintf('  Stepsize range: [%.4e, %.4e] (mean: %.4e)\n\n', ...
        min(output_adaptive_rgd.stepsize_history), max(output_adaptive_rgd.stepsize_history), ...
        mean(output_adaptive_rgd.stepsize_history));

%% Method 7: Adaptive PGD (PGD with Adaptive Stepsize, no preconditioner)
fprintf('--- Method 7: Adaptive PGD (with Line Search, no preconditioner) ---\n');
mu_adaptive_pgd = 0.3;  % Initial stepsize for adaptive PGD (same as fixed PGD)
fprintf('Using initial stepsize: μ = %.2f (will adapt via line search)\n', mu_adaptive_pgd);
params_adaptive_pgd = struct();
params_adaptive_pgd.T = T_refine;
params_adaptive_pgd.mu = mu_adaptive_pgd;  % Initial/default stepsize
params_adaptive_pgd.projection = @(X) rank_projection(X, r);
params_adaptive_pgd.use_preconditioner = false;  % NO preconditioner
params_adaptive_pgd.Xstar = Xstar;
% Enable adaptive stepsize
params_adaptive_pgd.use_adaptive_stepsize = true;
params_adaptive_pgd.line_search_max_iter = 20;
params_adaptive_pgd.line_search_beta = 0.5;
params_adaptive_pgd.line_search_c = 1e-4;

tic;
[output_adaptive_pgd, X_adaptive_pgd] = solve_PGD_amplitude(X0, [], y, operator, d, d, [], m, params_adaptive_pgd);
time_adaptive_pgd = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_adaptive_pgd, time_adaptive_pgd + init_time);
fprintf('  Initial error: %.6e\n', output_adaptive_pgd.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_adaptive_pgd.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_adaptive_pgd.Error_Stand(1) / output_adaptive_pgd.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_adaptive_pgd.Error_function(end));
fprintf('  Rank: %d\n', rank(X_adaptive_pgd, 1e-6));
fprintf('  Stepsize range: [%.4e, %.4e] (mean: %.4e)\n\n', ...
        min(output_adaptive_pgd.stepsize_history), max(output_adaptive_pgd.stepsize_history), ...
        mean(output_adaptive_pgd.stepsize_history));

%% Method 7: Adaptive PRGD with 1/2 Power (Preconditioned RGD with Adaptive Stepsize, power=1/2)
fprintf('--- Method 7: Adaptive PRGD with 1/2 Power (with Line Search, preconditioned, power=1/2) ---\n');
mu_adaptive_half_power = 0.3;  % Initial stepsize for adaptive PRGD with 1/2 power
fprintf('Using initial stepsize: μ = %.2f (will adapt via line search)\n', mu_adaptive_half_power);
params_adaptive_half_power = struct();
params_adaptive_half_power.T = T_refine;
params_adaptive_half_power.mu = mu_adaptive_half_power;  % Initial/default stepsize
params_adaptive_half_power.r = r;
params_adaptive_half_power.use_preconditioner = true;  % WITH preconditioner
params_adaptive_half_power.epsilon = 1e-8;
params_adaptive_half_power.preconditioner_power = 0.25;  % Use 1/2 power instead of 1/4
params_adaptive_half_power.Xstar = Xstar;
params_adaptive_half_power.use_spectral_init = false;  % Start from provided X0
params_adaptive_half_power.verbose = 0;
params_adaptive_half_power.return_factorized = false;
% Enable adaptive stepsize
params_adaptive_half_power.use_adaptive_stepsize = true;
params_adaptive_half_power.line_search_max_iter = 20;
params_adaptive_half_power.line_search_beta = 0.5;
params_adaptive_half_power.line_search_c = 1e-4;

tic;
[output_adaptive_half_power, X_adaptive_half_power] = solve_RGD_amplitude(X0, [], y, operator, d, d, [], m, params_adaptive_half_power);
time_adaptive_half_power = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_adaptive_half_power, time_adaptive_half_power + init_time);
fprintf('  Initial error: %.6e\n', output_adaptive_half_power.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_adaptive_half_power.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_adaptive_half_power.Error_Stand(1) / output_adaptive_half_power.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_adaptive_half_power.Error_function(end));
fprintf('  Rank: %d (factorized: %dx%d + %dx%d)\n', ...
        rank(X_adaptive_half_power, 1e-6), size(output_adaptive_half_power.U, 1), size(output_adaptive_half_power.U, 2), ...
        size(output_adaptive_half_power.V, 1), size(output_adaptive_half_power.V, 2));
fprintf('  Stepsize range: [%.4e, %.4e] (mean: %.4e)\n', ...
        min(output_adaptive_half_power.stepsize_history), max(output_adaptive_half_power.stepsize_history), ...
        mean(output_adaptive_half_power.stepsize_history));
fprintf('  Preconditioner power: 1/2\n\n');

%% Method 8: Adaptive PPGD (Preconditioned PGD with Adaptive Stepsize)
fprintf('--- Method 8: Adaptive PPGD (with Line Search, preconditioned) ---\n');
mu_adaptive_ppgd = 0.1;  % Initial stepsize for adaptive PPGD (same as fixed PPGD)
fprintf('Using initial stepsize: μ = %.2f (will adapt via line search)\n', mu_adaptive_ppgd);
params_adaptive_ppgd = struct();
params_adaptive_ppgd.T = T_refine;
params_adaptive_ppgd.mu = mu_adaptive_ppgd;  % Initial/default stepsize
params_adaptive_ppgd.projection = @(X) rank_projection(X, r);
params_adaptive_ppgd.use_preconditioner = true;  % WITH preconditioner
params_adaptive_ppgd.epsilon = 1e-8;
params_adaptive_ppgd.Xstar = Xstar;
% Enable adaptive stepsize
params_adaptive_ppgd.use_adaptive_stepsize = true;
params_adaptive_ppgd.line_search_max_iter = 20;
params_adaptive_ppgd.line_search_beta = 0.5;
params_adaptive_ppgd.line_search_c = 1e-4;

tic;
[output_adaptive_ppgd, X_adaptive_ppgd] = solve_PGD_amplitude(X0, [], y, operator, d, d, [], m, params_adaptive_ppgd);
time_adaptive_ppgd = toc;

fprintf('Results:\n');
fprintf('  Time: %.4f seconds (%.4f total with init)\n', time_adaptive_ppgd, time_adaptive_ppgd + init_time);
fprintf('  Initial error: %.6e\n', output_adaptive_ppgd.Error_Stand(1));
fprintf('  Final error: %.6e\n', output_adaptive_ppgd.Error_Stand(end));
fprintf('  Error reduction: %.2fx\n', output_adaptive_ppgd.Error_Stand(1) / output_adaptive_ppgd.Error_Stand(end));
fprintf('  Final loss: %.6e\n', output_adaptive_ppgd.Error_function(end));
fprintf('  Rank: %d\n', rank(X_adaptive_ppgd, 1e-6));
fprintf('  Stepsize range: [%.4e, %.4e] (mean: %.4e)\n\n', ...
        min(output_adaptive_ppgd.stepsize_history), max(output_adaptive_ppgd.stepsize_history), ...
        mean(output_adaptive_ppgd.stepsize_history));

%% Visualization and Comparison
figure('Position', [100, 100, 2400, 1000]);

% Plot 1: Error vs Iteration curves (all start from same initialization)
subplot(2, 7, 1);
semilogy(output_prgd.Error_Stand, 'b-', 'LineWidth', 2, 'DisplayName', 'PRGD (fixed)');
hold on;
semilogy(output_rgd.Error_Stand, 'r--', 'LineWidth', 2, 'DisplayName', 'RGD (fixed)');
semilogy(output_pgd.Error_Stand, 'k:', 'LineWidth', 2, 'DisplayName', 'PGD');
semilogy(output_ppgd.Error_Stand, 'g-.', 'LineWidth', 2, 'DisplayName', 'PPGD');
semilogy(output_adaptive.Error_Stand, 'm-.', 'LineWidth', 2, 'DisplayName', 'PRGD (adapt)');
semilogy(output_adaptive_rgd.Error_Stand, 'c:', 'LineWidth', 2, 'DisplayName', 'RGD (adapt)');
semilogy(output_adaptive_pgd.Error_Stand, 'Color', [0.8 0.4 0], 'LineWidth', 2, 'DisplayName', 'PGD (adapt)');
semilogy(output_adaptive_ppgd.Error_Stand, 'Color', [0.5 0 0.5], 'LineWidth', 2, 'DisplayName', 'PPGD (adapt)');
semilogy(output_adaptive_half_power.Error_Stand, 'Color', [0 0.7 0.7], 'LineWidth', 2, 'DisplayName', 'PRGD (1/2, adapt)');
yline(init_error, 'Color', [0.5 0.5 0.5], 'LineStyle', '-.', 'LineWidth', 1.5, 'DisplayName', 'Init Error');
hold off;
xlabel('Iteration');
ylabel('Relative Error');
title('Error vs Iteration (All Methods)');
legend('Location', 'best', 'FontSize', 6);
grid on;

% Plot 2: Error vs Time curves
subplot(2, 7, 2);
% Create time vectors for each method
time_prgd_vec = linspace(0, time_prgd, length(output_prgd.Error_Stand));
time_rgd_vec = linspace(0, time_rgd, length(output_rgd.Error_Stand));
time_pgd_vec = linspace(0, time_pgd, length(output_pgd.Error_Stand));
time_ppgd_vec = linspace(0, time_ppgd, length(output_ppgd.Error_Stand));
time_adaptive_vec = linspace(0, time_adaptive, length(output_adaptive.Error_Stand));
time_adaptive_rgd_vec = linspace(0, time_adaptive_rgd, length(output_adaptive_rgd.Error_Stand));
time_adaptive_pgd_vec = linspace(0, time_adaptive_pgd, length(output_adaptive_pgd.Error_Stand));
time_adaptive_ppgd_vec = linspace(0, time_adaptive_ppgd, length(output_adaptive_ppgd.Error_Stand));
time_adaptive_half_power_vec = linspace(0, time_adaptive_half_power, length(output_adaptive_half_power.Error_Stand));

semilogy(time_prgd_vec, output_prgd.Error_Stand, 'b-', 'LineWidth', 2, 'DisplayName', 'PRGD (fixed)');
hold on;
semilogy(time_rgd_vec, output_rgd.Error_Stand, 'r--', 'LineWidth', 2, 'DisplayName', 'RGD (fixed)');
semilogy(time_pgd_vec, output_pgd.Error_Stand, 'k:', 'LineWidth', 2, 'DisplayName', 'PGD');
semilogy(time_ppgd_vec, output_ppgd.Error_Stand, 'g-.', 'LineWidth', 2, 'DisplayName', 'PPGD');
semilogy(time_adaptive_vec, output_adaptive.Error_Stand, 'm-.', 'LineWidth', 2, 'DisplayName', 'PRGD (adapt)');
semilogy(time_adaptive_rgd_vec, output_adaptive_rgd.Error_Stand, 'c:', 'LineWidth', 2, 'DisplayName', 'RGD (adapt)');
semilogy(time_adaptive_pgd_vec, output_adaptive_pgd.Error_Stand, 'Color', [0.8 0.4 0], 'LineWidth', 2, 'DisplayName', 'PGD (adapt)');
semilogy(time_adaptive_ppgd_vec, output_adaptive_ppgd.Error_Stand, 'Color', [0.5 0 0.5], 'LineWidth', 2, 'DisplayName', 'PPGD (adapt)');
semilogy(time_adaptive_half_power_vec, output_adaptive_half_power.Error_Stand, 'Color', [0 0.7 0.7], 'LineWidth', 2, 'DisplayName', 'PRGD (1/2, adapt)');
hold off;
xlabel('Time (seconds)');
ylabel('Relative Error');
title('Error vs Time');
legend('Location', 'best', 'FontSize', 6);
grid on;

% Plot 3: Loss curves
subplot(2, 7, 3);
semilogy(output_prgd.Error_function, 'b-', 'LineWidth', 2, 'DisplayName', 'PRGD (fixed)');
hold on;
semilogy(output_rgd.Error_function, 'r--', 'LineWidth', 2, 'DisplayName', 'RGD (fixed)');
semilogy(output_pgd.Error_function, 'k:', 'LineWidth', 2, 'DisplayName', 'PGD');
semilogy(output_ppgd.Error_function, 'g-.', 'LineWidth', 2, 'DisplayName', 'PPGD');
semilogy(output_adaptive.Error_function, 'm-.', 'LineWidth', 2, 'DisplayName', 'PRGD (adapt)');
semilogy(output_adaptive_rgd.Error_function, 'c:', 'LineWidth', 2, 'DisplayName', 'RGD (adapt)');
semilogy(output_adaptive_pgd.Error_function, 'Color', [0.8 0.4 0], 'LineWidth', 2, 'DisplayName', 'PGD (adapt)');
semilogy(output_adaptive_ppgd.Error_function, 'Color', [0.5 0 0.5], 'LineWidth', 2, 'DisplayName', 'PPGD (adapt)');
semilogy(output_adaptive_half_power.Error_function, 'Color', [0 0.7 0.7], 'LineWidth', 2, 'DisplayName', 'PRGD (1/2, adapt)');
hold off;
xlabel('Iteration');
ylabel('Loss Value');
title('Loss Function vs Iteration');
legend('Location', 'best', 'FontSize', 6);
grid on;

% Plot 4: Stepsize history for Adaptive PRGD (1/4 power)
subplot(2, 7, 4);
plot(output_adaptive.stepsize_history, 'm-', 'LineWidth', 1.5);
hold on;
yline(mu_adaptive, 'b--', 'Initial α', 'LineWidth', 1.5);
hold off;
xlabel('Iteration');
ylabel('Stepsize α');
title('Adaptive PRGD (1/4) Stepsize');
grid on;
legend('Adaptive α', 'Initial α', 'Location', 'best', 'FontSize', 7);

% Plot 5: Stepsize history for Adaptive RGD
subplot(2, 7, 5);
plot(output_adaptive_rgd.stepsize_history, 'c-', 'LineWidth', 1.5);
hold on;
yline(mu_adaptive_rgd, 'r--', 'Initial α', 'LineWidth', 1.5);
hold off;
xlabel('Iteration');
ylabel('Stepsize α');
title('Adaptive RGD Stepsize');
grid on;
legend('Adaptive α', 'Initial α', 'Location', 'best', 'FontSize', 7);

% Plot 6: Stepsize history for Adaptive PGD
subplot(2, 7, 6);
plot(output_adaptive_pgd.stepsize_history, 'Color', [0.8 0.4 0], 'LineWidth', 1.5);
hold on;
yline(mu_adaptive_pgd, 'k--', 'Initial α', 'LineWidth', 1.5);
hold off;
xlabel('Iteration');
ylabel('Stepsize α');
title('Adaptive PGD Stepsize');
grid on;
legend('Adaptive α', 'Initial α', 'Location', 'best', 'FontSize', 7);

% Plot 7: Stepsize history for Adaptive PRGD (1/2 power)
subplot(2, 7, 7);
plot(output_adaptive_half_power.stepsize_history, 'Color', [0 0.7 0.7], 'LineWidth', 1.5);
hold on;
yline(mu_adaptive_half_power, 'g--', 'Initial α', 'LineWidth', 1.5);
hold off;
xlabel('Iteration');
ylabel('Stepsize α');
title('Adaptive PRGD (1/2) Stepsize');
grid on;
legend('Adaptive α', 'Initial α', 'Location', 'best', 'FontSize', 7);

% Plot 8: Preconditioner comparison (adaptive stepsize: 1/4 vs 1/2)
subplot(2, 7, 8);
semilogy(output_adaptive.Error_Stand, 'm-.', 'LineWidth', 2, 'DisplayName', 'PRGD (1/4, adapt)');
hold on;
semilogy(output_adaptive_half_power.Error_Stand, 'Color', [0 0.7 0.7], 'LineWidth', 2, 'DisplayName', 'PRGD (1/2, adapt)');
hold off;
xlabel('Iteration');
ylabel('Relative Error');
title('Preconditioner Power Comparison');
legend('Location', 'best', 'FontSize', 7);
grid on;

fprintf('Figure: Comparison of 9 optimization methods (2×7 layout)\n');

sgtitle('Local Refinement Comparison (Same Initialization)');

%% Summary
fprintf('\n=== Summary Table ===\n');
fprintf('Method          | Refine Time (s) | Total Time (s) | Init Error  | Final Error | Error Reduction | Final Loss\n');
fprintf('----------------|-----------------|----------------|-------------|-------------|-----------------|------------\n');
fprintf('Init            | %15s | %14.4f | %.6e | %.6e | %15s | %s\n', '-', init_time, init_error, init_error, '1.00x', '-');
fprintf('PRGD (fixed)    | %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_prgd, time_prgd + init_time, output_prgd.Error_Stand(1), output_prgd.Error_Stand(end), ...
        output_prgd.Error_Stand(1) / output_prgd.Error_Stand(end), output_prgd.Error_function(end));
fprintf('RGD (fixed)     | %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_rgd, time_rgd + init_time, output_rgd.Error_Stand(1), output_rgd.Error_Stand(end), ...
        output_rgd.Error_Stand(1) / output_rgd.Error_Stand(end), output_rgd.Error_function(end));
fprintf('PGD             | %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_pgd, time_pgd + init_time, output_pgd.Error_Stand(1), output_pgd.Error_Stand(end), ...
        output_pgd.Error_Stand(1) / output_pgd.Error_Stand(end), output_pgd.Error_function(end));
fprintf('PPGD            | %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_ppgd, time_ppgd + init_time, output_ppgd.Error_Stand(1), output_ppgd.Error_Stand(end), ...
        output_ppgd.Error_Stand(1) / output_ppgd.Error_Stand(end), output_ppgd.Error_function(end));
fprintf('PRGD (adapt,1/4)| %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_adaptive, time_adaptive + init_time, output_adaptive.Error_Stand(1), output_adaptive.Error_Stand(end), ...
        output_adaptive.Error_Stand(1) / output_adaptive.Error_Stand(end), output_adaptive.Error_function(end));
fprintf('RGD (adapt)     | %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_adaptive_rgd, time_adaptive_rgd + init_time, output_adaptive_rgd.Error_Stand(1), output_adaptive_rgd.Error_Stand(end), ...
        output_adaptive_rgd.Error_Stand(1) / output_adaptive_rgd.Error_Stand(end), output_adaptive_rgd.Error_function(end));
fprintf('PGD (adapt)     | %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_adaptive_pgd, time_adaptive_pgd + init_time, output_adaptive_pgd.Error_Stand(1), output_adaptive_pgd.Error_Stand(end), ...
        output_adaptive_pgd.Error_Stand(1) / output_adaptive_pgd.Error_Stand(end), output_adaptive_pgd.Error_function(end));
fprintf('PPGD (adapt)    | %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_adaptive_ppgd, time_adaptive_ppgd + init_time, output_adaptive_ppgd.Error_Stand(1), output_adaptive_ppgd.Error_Stand(end), ...
        output_adaptive_ppgd.Error_Stand(1) / output_adaptive_ppgd.Error_Stand(end), output_adaptive_ppgd.Error_function(end));
fprintf('PRGD (adapt,1/2)| %15.4f | %14.4f | %.6e | %.6e | %15.2fx | %.6e\n', ...
        time_adaptive_half_power, time_adaptive_half_power + init_time, output_adaptive_half_power.Error_Stand(1), output_adaptive_half_power.Error_Stand(end), ...
        output_adaptive_half_power.Error_Stand(1) / output_adaptive_half_power.Error_Stand(end), output_adaptive_half_power.Error_function(end));
fprintf('\n');

fprintf('Stepsize Settings:\n');
fprintf('  RGD (fixed):      μ = %.2f (fixed, no preconditioner, factorized)\n', mu_rgd);
fprintf('  PGD:              μ = %.2f (fixed, no preconditioner, full matrix)\n', mu_pgd);
fprintf('  PRGD (adapt,1/4): μ = %.2f (initial, adapts via line search, preconditioned, power=1/4)\n', mu_adaptive);
fprintf('  RGD (adaptive):   μ = %.2f (initial, adapts via line search, no preconditioner)\n', mu_adaptive_rgd);
fprintf('  PGD (adaptive):   μ = %.2f (initial, adapts via line search, no preconditioner)\n', mu_adaptive_pgd);
fprintf('  PPGD (adaptive):  μ = %.2f (initial, adapts via line search, preconditioned)\n', mu_adaptive_ppgd);
fprintf('  PRGD (adapt,1/2): μ = %.2f (initial, adapts via line search, preconditioned, power=1/2)\n', mu_adaptive_half_power);
fprintf('\n');

fprintf('Memory Efficiency (Factorized Methods):\n');
fprintf('  PRGD stores: %d elements (%.1f%% of full %d×%d matrix)\n', ...
        numel(output_prgd.U) + numel(output_prgd.Sigma) + numel(output_prgd.V), ...
        100 * (numel(output_prgd.U) + numel(output_prgd.Sigma) + numel(output_prgd.V)) / (d*d), d, d);
fprintf('  Memory reduction: %.1fx smaller than full matrix\n', ...
        d*d / (numel(output_prgd.U) + numel(output_prgd.Sigma) + numel(output_prgd.V)));
fprintf('\n');

fprintf('Adaptive Stepsize Statistics:\n');
fprintf('  PRGD (adapt, 1/4 power):\n');
fprintf('    Stepsize range: [%.4e, %.4e]\n', ...
        min(output_adaptive.stepsize_history), max(output_adaptive.stepsize_history));
fprintf('    Mean stepsize: %.4e (initial: %.4e)\n', ...
        mean(output_adaptive.stepsize_history), mu_adaptive);
fprintf('    Std deviation: %.4e\n', std(output_adaptive.stepsize_history));
fprintf('  RGD (adaptive):\n');
fprintf('    Stepsize range: [%.4e, %.4e]\n', ...
        min(output_adaptive_rgd.stepsize_history), max(output_adaptive_rgd.stepsize_history));
fprintf('    Mean stepsize: %.4e (initial: %.4e)\n', ...
        mean(output_adaptive_rgd.stepsize_history), mu_adaptive_rgd);
fprintf('    Std deviation: %.4e\n', std(output_adaptive_rgd.stepsize_history));
fprintf('  PGD (adaptive):\n');
fprintf('    Stepsize range: [%.4e, %.4e]\n', ...
        min(output_adaptive_pgd.stepsize_history), max(output_adaptive_pgd.stepsize_history));
fprintf('    Mean stepsize: %.4e (initial: %.4e)\n', ...
        mean(output_adaptive_pgd.stepsize_history), mu_adaptive_pgd);
fprintf('    Std deviation: %.4e\n', std(output_adaptive_pgd.stepsize_history));
fprintf('  PPGD (adaptive):\n');
fprintf('    Stepsize range: [%.4e, %.4e]\n', ...
        min(output_adaptive_ppgd.stepsize_history), max(output_adaptive_ppgd.stepsize_history));
fprintf('    Mean stepsize: %.4e (initial: %.4e)\n', ...
        mean(output_adaptive_ppgd.stepsize_history), mu_adaptive_ppgd);
fprintf('    Std deviation: %.4e\n', std(output_adaptive_ppgd.stepsize_history));
fprintf('  PRGD (adapt, 1/2 power):\n');
fprintf('    Stepsize range: [%.4e, %.4e]\n', ...
        min(output_adaptive_half_power.stepsize_history), max(output_adaptive_half_power.stepsize_history));
fprintf('    Mean stepsize: %.4e (initial: %.4e)\n', ...
        mean(output_adaptive_half_power.stepsize_history), mu_adaptive_half_power);
fprintf('    Std deviation: %.4e\n', std(output_adaptive_half_power.stepsize_history));
fprintf('\n');

fprintf('Relative Performance:\n');
fprintf('Preconditioner Effect (Fixed Stepsize):\n');
if output_rgd.Error_Stand(end) > 0 && output_prgd.Error_Stand(end) > 0
    improvement = output_rgd.Error_Stand(end) / output_prgd.Error_Stand(end);
    fprintf('  PRGD (fixed) achieves %.2fx better final error than RGD (fixed)\n', improvement);
end

fprintf('Adaptive Stepsize Effect (PRGD):\n');
if output_prgd.Error_Stand(end) > 0 && output_adaptive.Error_Stand(end) > 0
    improvement = output_prgd.Error_Stand(end) / output_adaptive.Error_Stand(end);
    if improvement > 1
        fprintf('  PRGD (adaptive) achieves %.2fx better error than PRGD (fixed)\n', improvement);
    else
        fprintf('  PRGD (fixed) achieves %.2fx better error than PRGD (adaptive)\n', 1/improvement);
    end
end

fprintf('Adaptive Stepsize Effect (RGD):\n');
if output_rgd.Error_Stand(end) > 0 && output_adaptive_rgd.Error_Stand(end) > 0
    improvement = output_rgd.Error_Stand(end) / output_adaptive_rgd.Error_Stand(end);
    if improvement > 1
        fprintf('  RGD (adaptive) achieves %.2fx better error than RGD (fixed)\n', improvement);
    else
        fprintf('  RGD (fixed) achieves %.2fx better error than RGD (adaptive)\n', 1/improvement);
    end
end

fprintf('Preconditioner Effect (Fixed Stepsize):\n');
if output_pgd.Error_Stand(end) > 0 && output_ppgd.Error_Stand(end) > 0
    improvement = output_pgd.Error_Stand(end) / output_ppgd.Error_Stand(end);
    fprintf('  PPGD achieves %.2fx better final error than PGD\n', improvement);
end
fprintf('\n');

fprintf('Preconditioner Effect (Adaptive Stepsize):\n');
if output_adaptive_pgd.Error_Stand(end) > 0 && output_adaptive_ppgd.Error_Stand(end) > 0
    improvement = output_adaptive_pgd.Error_Stand(end) / output_adaptive_ppgd.Error_Stand(end);
    fprintf('  PPGD (adaptive) achieves %.2fx better final error than PGD (adaptive)\n', improvement);
end
fprintf('\n');

fprintf('Adaptive Stepsize Effect (PGD):\n');
if output_pgd.Error_Stand(end) > 0 && output_adaptive_pgd.Error_Stand(end) > 0
    improvement = output_pgd.Error_Stand(end) / output_adaptive_pgd.Error_Stand(end);
    if improvement > 1
        fprintf('  PGD (adaptive) achieves %.2fx better error than PGD (fixed)\n', improvement);
    else
        fprintf('  PGD (fixed) achieves %.2fx better error than PGD (adaptive)\n', 1/improvement);
    end
end

fprintf('Adaptive Stepsize Effect (PPGD):\n');
if output_ppgd.Error_Stand(end) > 0 && output_adaptive_ppgd.Error_Stand(end) > 0
    improvement = output_ppgd.Error_Stand(end) / output_adaptive_ppgd.Error_Stand(end);
    if improvement > 1
        fprintf('  PPGD (adaptive) achieves %.2fx better error than PPGD (fixed)\n', improvement);
    else
        fprintf('  PPGD (fixed) achieves %.2fx better error than PPGD (adaptive)\n', 1/improvement);
    end
end
fprintf('\n');

fprintf('Best Overall:\n');
all_errors = [output_prgd.Error_Stand(end), output_rgd.Error_Stand(end), output_pgd.Error_Stand(end), ...
              output_ppgd.Error_Stand(end), output_adaptive.Error_Stand(end), output_adaptive_rgd.Error_Stand(end), ...
              output_adaptive_pgd.Error_Stand(end), output_adaptive_ppgd.Error_Stand(end)];
method_names = {'PRGD (fixed)', 'RGD (fixed)', 'PGD', 'PPGD', 'PRGD (adaptive)', 'RGD (adaptive)', ...
                'PGD (adaptive)', 'PPGD (adaptive)'};
[min_error, min_idx] = min(all_errors);
fprintf('  Best method: %s with error %.6e\n', method_names{min_idx}, min_error);

fprintf('\nTest completed successfully!\n');
