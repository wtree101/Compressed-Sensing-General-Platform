%% Test Tensor Nuclear Norm Initialization + Local Refinement
% This script tests the complete pipeline:
% 1. Initialize using tensor nuclear norm minimization (convex, few iterations)
% 2. Refine using gradient descent on matrix manifold (non-convex, many iterations)
%
% This mimics the workflow in Phasediagram_tensor.m

clear; clc;

fprintf('=== Test: Tensor Nuclear Norm Initialization + Local Refinement ===\n\n');

%% Problem Setup
d = 20;              % Dimension
r = 2;               % True rank
m = 200;             % Number of measurements
kappa = 2;           % Condition number

fprintf('Problem Configuration:\n');
fprintf('  Dimension: d=%d\n', d);
fprintf('  True rank: r=%d\n', r);
fprintf('  Measurements: m=%d (%.1fx overdetermined)\n', m, m/(d*r));
fprintf('  Condition number: kappa=%.1f\n\n', kappa);

%% Generate Ground Truth
fprintf('Generating ground truth...\n');
%rng(42);  % For reproducibility

U_true = randn(d, r);
U_true = orth(U_true);
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
    A_cells{i} = (Ai + Ai') / 2;  % Symmetrize
end

operator = struct();
operator.A_cells = A_cells;

% Matrix operator for local refinement
A_mat = zeros(m, d*d);
for i = 1:m
    A_mat(i, :) = A_cells{i}(:)';
end
operator.A = @(X) A_mat * X(:);
operator.A_star = @(y) reshape(A_mat' * y, [d, d]);

fprintf('  Created %d measurement matrices\n\n', m);

%% Generate Measurements (Phase Retrieval Model)
fprintf('Generating measurements (phase retrieval)...\n');

% y_i = |<A_i, X>|² for phase retrieval
y = zeros(m, 1);
for i = 1:m
    y(i) = abs(trace(A_cells{i}' * Xstar))^2 / sqrt(m);
end

fprintf('  Measurement vector norm: %.6f\n\n', norm(y));

%% Stage 1: Initialization using Tensor Nuclear Norm
fprintf('=== Stage 1: Tensor Nuclear Norm Initialization ===\n\n');

% Initialization parameters
init_max_iter = 10;   % Few iterations for initialization
init_lambda = 1.0;
init_rho = 0.1;
init_verbose = 1;

fprintf('Initialization parameters:\n');
fprintf('  Max iterations: %d\n', init_max_iter);
fprintf('  Lambda: %.2f\n', init_lambda);
fprintf('  Rho: %.2f\n\n', init_rho);

% Run initialization
tic;

% Prepare initialization parameters
init_params = struct();
init_params.rank = r;
init_params.max_iter = init_max_iter;
init_params.lambda = init_lambda;
init_params.rho = init_rho;
init_params.verbose = init_verbose;
init_params.normalize = true;

% Call with unified signature: [X0, U0, history] = func(y, operator, d1, d2, params)
[X0, ~, init_history] = initialize_tensor_nuclear_norm(y, operator, d, d, init_params);
init_time = toc;

% Evaluate initialization
[init_error, X0_aligned] = rectify_sign_ambiguity(X0, Xstar);

fprintf('\nInitialization Results:\n');
fprintf('  Time: %.4f seconds\n', init_time);
fprintf('  Reconstruction error: %.6e\n', init_error);
fprintf('  Recovered rank: %d\n', rank(X0, 1e-6));
fprintf('  Symmetry error: %.6e\n\n', norm(X0 - X0', 'fro'));

%% Stage 2: Local Refinement using Gradient Descent
fprintf('=== Stage 2: Local Refinement (Gradient Descent) ===\n\n');

% Refinement parameters
refine_max_iter = 500;
mu = 0.1;  % Step size
verbose_refine = 1;

fprintf('Refinement parameters:\n');
fprintf('  Max iterations: %d\n', refine_max_iter);
fprintf('  Step size (mu): %.4f\n\n', mu);

% Prepare parameters for local solver
trial_params = struct();
trial_params.d1 = d;
trial_params.d2 = d;
trial_params.m = m;
trial_params.r = r;
trial_params.kappa = kappa;
trial_params.T = refine_max_iter;
trial_params.Xstar = Xstar;
trial_params.verbose = verbose_refine;
trial_params.X0 = X0_aligned;  % Use initialized matrix
trial_params.operator = operator;
trial_params.y = y;

% Run local refinement using solve_PGD_amplitude
tic;

% Prepare parameters for solve_PGD_amplitude
refine_params = struct();
refine_params.T = refine_max_iter;
refine_params.mu = mu;
refine_params.Xstar = Xstar;
refine_params.projection = @(X) project_rank_r(X, r);  % Rank-r projection

% Call solver with correct signature: solve_PGD_amplitude(Xl, ~, y, operator, d1, d2, ~, m, params)
[output, X_final] = solve_PGD_amplitude(X0_aligned, [], y, operator, d, d, [], m, refine_params);
refine_time = toc;

% Evaluate final result
[final_error, X_final_aligned] = rectify_sign_ambiguity(X_final, Xstar);

fprintf('\nRefinement Results:\n');
fprintf('  Time: %.4f seconds\n', refine_time);
fprintf('  Final reconstruction error: %.6e\n', final_error);
fprintf('  Recovered rank: %d\n', rank(X_final, 1e-6));
fprintf('  Symmetry error: %.6e\n\n', norm(X_final - X_final', 'fro'));

%% Comparison with Random Initialization
fprintf('=== Comparison: Random Initialization ===\n\n');

% Random initialization
X0_random = randn(d, d);
X0_random = (X0_random + X0_random') / 2;
X0_random = X0_random / norm(X0_random, 'fro');

fprintf('Random initialization:\n');
[random_init_error, ~] = rectify_sign_ambiguity(X0_random, Xstar);
fprintf('  Initial error: %.6e\n', random_init_error);

% Run refinement from random init
tic;

% Prepare parameters for random initialization
random_params = struct();
random_params.T = refine_max_iter;
random_params.mu = mu;
random_params.Xstar = Xstar;
random_params.projection = @(X) project_rank_r(X, r);  % Rank-r projection

% Call solver with correct signature
[output_random, X_random_final] = solve_PGD_amplitude(X0_random, [], y, operator, d, d, [], m, random_params);
random_refine_time = toc;

[random_final_error, ~] = rectify_sign_ambiguity(X_random_final, Xstar);

fprintf('  Time: %.4f seconds\n', random_refine_time);
fprintf('  Final error: %.6e\n\n', random_final_error);

%% Visualization
fprintf('=== Visualization ===\n');

% Create figure with subplots
figure('Position', [100, 100, 1400, 800]);

% Plot 1: Ground truth
subplot(2, 3, 1);
imagesc(Xstar);
colorbar;
axis square;
title(sprintf('Ground Truth (rank=%d)', rank(Xstar, 1e-6)));
xlabel('Column');
ylabel('Row');

% Plot 2: Tensor nuclear norm initialization
subplot(2, 3, 2);
imagesc(X0_aligned);
colorbar;
axis square;
title(sprintf('TNN Init (err=%.2e)', init_error));
xlabel('Column');
ylabel('Row');

% Plot 3: Final result after refinement
subplot(2, 3, 3);
imagesc(X_final_aligned);
colorbar;
axis square;
title(sprintf('Final (TNN + Refine, err=%.2e)', final_error));
xlabel('Column');
ylabel('Row');

% % Plot 4: Convergence comparison
% subplot(2, 3, 4);
% semilogy(1:length(output), output, 'b-', 'LineWidth', 2, 'DisplayName', 'TNN Init');
% hold on;
% semilogy(1:length(output_random), output_random, 'r--', 'LineWidth', 2, 'DisplayName', 'Random Init');
% xlabel('Iteration');
% ylabel('Reconstruction Error');
% title('Convergence Comparison');
% legend('Location', 'best');
% grid on;

% Plot 5: Error difference (TNN init vs final)
subplot(2, 3, 5);
imagesc(abs(X_final_aligned - X0_aligned));
colorbar;
axis square;
title('|Final - Init|');
xlabel('Column');
ylabel('Row');

% Plot 6: Error vs ground truth
subplot(2, 3, 6);
imagesc(abs(X_final_aligned - Xstar));
colorbar;
axis square;
title(sprintf('|Final - Truth| (err=%.2e)', final_error));
xlabel('Column');
ylabel('Row');

sgtitle('Tensor Nuclear Norm Init + Local Refinement');

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('\nInitialization Comparison:\n');
fprintf('  TNN Init error:    %.6e\n', init_error);
fprintf('  Random Init error: %.6e\n', random_init_error);
fprintf('  Improvement:       %.2fx better\n', random_init_error / init_error);

fprintf('\nFinal Results:\n');
fprintf('  TNN + Refine:      %.6e\n', final_error);
fprintf('  Random + Refine:   %.6e\n', random_final_error);
fprintf('  Improvement:       %.2fx better\n', random_final_error / final_error);

fprintf('\nTiming:\n');
fprintf('  TNN Init:          %.4f seconds\n', init_time);
fprintf('  Local Refine:      %.4f seconds\n', refine_time);
fprintf('  Total (TNN+Refine):%.4f seconds\n', init_time + refine_time);
fprintf('  Random+Refine:     %.4f seconds\n', random_refine_time);

fprintf('\nConclusion:\n');
if final_error < random_final_error * 0.5
    fprintf('  ✓ TNN initialization provides SIGNIFICANT improvement\n');
elseif final_error < random_final_error
    fprintf('  ✓ TNN initialization provides modest improvement\n');
else
    fprintf('  ~ TNN initialization does not improve over random\n');
end

if final_error < 1e-3
    fprintf('  ✓ Final reconstruction is EXCELLENT (error < 1e-3)\n');
elseif final_error < 1e-2
    fprintf('  ✓ Final reconstruction is GOOD (error < 1e-2)\n');
else
    fprintf('  ~ Final reconstruction needs improvement\n');
end

fprintf('\n=== Test Complete ===\n');

%% Helper Function
function X_proj = project_rank_r(X, r)
    % Project matrix X onto the set of symmetric rank-r matrices
    % Input:
    %   X - Input matrix (d x d)
    %   r - Target rank
    % Output:
    %   X_proj - Projected symmetric rank-r matrix
    
    % Symmetrize
    X_sym = (X + X') / 2;
    
    % SVD and truncate to rank r
    [U, S, ~] = svd(X_sym);
    U_r = U(:, 1:r);
    S_r = S(1:r, 1:r);
    
    % Reconstruct symmetric rank-r matrix
    X_proj = U_r * S_r * U_r';
end
