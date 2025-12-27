%% Test Spectral Initialization + RGD Refinement
% Test two-stage approach:
%   1. Initialize using TuckerTensor.initialize_spectral (direct tensor formation)
%   2. Refine using solve_RGD_tucker_kronecker (T steps)
%
% This uses the spectral initialization method that:
%   - Forms T = sum_i y_i * (Ai ⊗ Ai) directly
%   - Applies HOSVD_with_factors to extract factor matrices
%   - Computes core tensor via tensor_mode_product

clear; clc;

fprintf('=== Spectral Initialization + RGD Refinement Test (Non-Symmetric) ===\n\n');

%% Setup
d1 = 20; d2 = 30;  % Non-square matrix dimensions
r = 3; r_tucker = r; m = 600; 
mu_rgd = 0.1;   % Step size for RGD refinement
T_rgd = 200;    % Number of RGD refinement iterations
use_preprocessing = false;  % Set to false to disable preprocessing
rng(47);

% Ground truth (non-symmetric matrix)
U1_true = randn(d1, r);
U2_true = randn(d2, r);
Xstar = abs(U1_true) * abs(U2_true)';  % d1 × d2 matrix
Xstar = Xstar / norm(Xstar, 'fro');

% Create measurement operator (phase retrieval measurements)
n = d1 * d2;
A = randn(m, n);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d1, d2]);

% Generate measurements: phase retrieval (magnitude only)
y = abs(operator.A(Xstar)) / sqrt(m);

% Define preprocessing function for measurements
if use_preprocessing
    pre_func = @(y) set_zero_outside_range_tensor(y);
else
    pre_func = @(y) y;  % Identity function (no preprocessing)
end

fprintf('Setup: d1=%d, d2=%d, r=%d, r_tucker=%d, m=%d\n', d1, d2, r, r_tucker, m);
fprintf('Matrix type: Non-Symmetric (no symmetrization)\n');
if use_preprocessing
    fprintf('Preprocessing: set_zero_outside_range enabled\n');
else
    fprintf('Preprocessing: DISABLED\n');
end
fprintf('RGD iterations: %d\n', T_rgd);

% Display measurement statistics
fprintf('\nMeasurement statistics (before preprocessing):\n');
fprintf('  Mean: %.4f, Std: %.4f\n', mean(y), std(y));
fprintf('  Min:  %.4f, Max: %.4f\n', min(y), max(y));
fprintf('  Expected range: [%.4f, %.4f]\n', 1/sqrt(m), 5/sqrt(m));
fprintf('\n');

%% Step 1: Extract measurement matrices and create TuckerOperator
fprintf('=== Step 1: Creating TuckerOperator ===\n');

A_matrix_extracted = zeros(m, n);
for j = 1:n
    e_j = zeros(n, 1);
    e_j(j) = 1;
    E_j = reshape(e_j, [d1, d2]);
    A_matrix_extracted(:, j) = operator.A(E_j);
end

% Create cell array of measurement matrices
A_cells = cell(m, 1);
for i = 1:m
    Ai = reshape(A_matrix_extracted(i, :), [d1, d2]);
    A_cells{i} = Ai;  % No symmetrization
end

% Create Tucker operator
tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false);
tucker_op.A_mat = A_matrix_extracted';  % Store for efficient computation (n × m)

fprintf('TuckerOperator created with %d measurement matrices.\n\n', m);

%% Step 2: Spectral Initialization using TuckerTensor.initialize_spectral
fprintf('=== Step 2: Spectral Initialization ===\n');

% Create TuckerTensor object for initialization (non-square: d1 x d2 x d1 x d2)
dims = [d1, d2, d1, d2];
T_tucker_init = TuckerTensor(dims, r_tucker, 'symmetric', false, 'init_method', 'zeros');

% Prepare operator struct for spectral initialization
spectral_operator = struct();
spectral_operator.A_cells = A_cells;

% Convert phase retrieval measurements to tensor measurements
% For spectral init: we need y_i representing <Ai ⊗ Ai, T>
% Since y_i = |<Ai, X>| / sqrt(m) and T = X ⊗ X
% We have: <Ai ⊗ Ai, X ⊗ X> = (<Ai, X>)^2
% So: y_spectral_i = (<Ai, X>)^2 / sqrt(m) = (y_i * sqrt(m))^2 / sqrt(m) = y_i^2 * sqrt(m)
y_spectral = y.^2 * sqrt(m);

fprintf('Measurements converted: phase retrieval -> tensor measurements\n');
fprintf('Running spectral initialization...\n\n');

tic;
% Pass pre_func as parameter to initialize_spectral
% The function will apply it internally before forming the tensor
[U_cell_init, G_init] = T_tucker_init.initialize_spectral(spectral_operator, y_spectral, m, 'pre_func', pre_func);
time_init = toc;

% For reporting purposes, calculate how many measurements were kept
y_preprocessed = pre_func(y_spectral);

% Set the initialized factor matrices and core
for k = 1:4
    T_tucker_init.U{k} = U_cell_init{k};
end
T_tucker_init.G = G_init;

% Extract matrix for error computation
X_init = extract_matrix_from_tucker_2(T_tucker_init);
% No symmetrization

[error_init, ~] = rectify_sign_ambiguity(X_init, Xstar);

fprintf('Spectral initialization complete:\n');
fprintf('  Error: %.6e\n', error_init);
fprintf('  Time:  %.2f seconds\n', time_init);
fprintf('  Core norm: %.6f\n', norm(G_init(:)));

% Display factor matrix info
fprintf('  Factor matrix U: (%d x %d)\n', size(U_cell_init{1}, 1), size(U_cell_init{1}, 2));
fprintf('  Factor matrix condition number: %.2e\n', cond(U_cell_init{1}));
fprintf('\n');

%% Step 3: RGD Refinement
fprintf('=== Step 3: RGD Refinement (%d iterations) ===\n', T_rgd);

% y_spectral is already in the correct format for RGD
params_rgd = struct('T', T_rgd, 'mu', mu_rgd, 'Xstar', Xstar, 'verbose', 1);

tic;
[output_rgd, T_tucker_final] = solve_RGD_tucker_kronecker(T_tucker_init, [], y_spectral, tucker_op, [], [], [], m, params_rgd);
time_rgd = toc;

% Extract final matrix
X_final = extract_matrix_from_tucker_2(T_tucker_final);
% No symmetrization

[error_final, ~] = rectify_sign_ambiguity(X_final, Xstar);

fprintf('RGD refinement complete: Error=%.2e, Time=%.2fs\n', error_final, time_rgd);
fprintf('Final loss: %.2e\n\n', output_rgd.Error_function(end));

%% Step 4: Results Summary
fprintf('=== Results Summary ===\n');
fprintf('Spectral Initialization:\n');
fprintf('  Error: %.6e\n', error_init);
fprintf('  Time:  %.2f seconds\n', time_init);

fprintf('\nRGD Refinement:\n');
fprintf('  Error: %.6e (after %d iterations)\n', error_final, T_rgd);
fprintf('  Loss:  %.6e\n', output_rgd.Error_function(end));
fprintf('  Time:  %.2f seconds\n', time_rgd);

fprintf('\nOverall Improvement:\n');
improvement = (error_init - error_final) / error_init * 100;
fprintf('  Error reduction: %.2f%%\n', improvement);

fprintf('\nTotal time: %.2f seconds\n', time_init + time_rgd);

if use_preprocessing
    fprintf('\nPreprocessing Impact:\n');
    fprintf('  Measurements removed: %d/%d (%.1f%%)\n', ...
            sum(y_preprocessed == 0), m, sum(y_preprocessed == 0)/m*100);
    fprintf('  Measurements kept: %d/%d (%.1f%%)\n', ...
            sum(y_preprocessed ~= 0), m, sum(y_preprocessed ~= 0)/m*100);
end
fprintf('\n');

%% Step 5: Convergence Visualization
fprintf('=== Generating Convergence Plots ===\n');

figure('Position', [100, 100, 1400, 500]);

% Plot 1: Loss convergence (RGD only, spectral init has no iterative loss)
subplot(1, 3, 1);
semilogy(output_rgd.Error_function, 'r-', 'LineWidth', 2);
xlabel('RGD Iteration'); ylabel('Loss'); 
title('RGD Loss Convergence'); grid on;

% Plot 2: Matrix error convergence (RGD)
subplot(1, 3, 2);
semilogy(output_rgd.Error_Stand, 'r-', 'LineWidth', 2);
yline(error_init, 'b--', 'Spectral Init', 'LineWidth', 1.5);
xlabel('RGD Iteration'); ylabel('Matrix Error'); 
title('Error Convergence'); grid on;
legend('RGD', 'Spectral Init', 'Location', 'best');

% Plot 3: Error comparison bar chart
subplot(1, 3, 3);
errors = [error_init, error_final];
bar(errors);
set(gca, 'XTickLabel', {'Spectral Init', 'After RGD'});
ylabel('Relative Error');
title('Error Comparison');
grid on;
%yscale log;

if use_preprocessing
    preproc_str = 'ON';
else
    preproc_str = 'OFF';
end
sgtitle(sprintf('Spectral Init + RGD (%dx%d, r=%d, m=%d, T=%d) - No Symmetrization, Preproc: %s', ...
        d1, d2, r, m, T_rgd, preproc_str));

fprintf('Convergence plots generated.\n\n');

%% Memory Analysis
full_mem = (d1*d2)^2 * 8;
tucker_mem = (2*d1*r_tucker + 2*d2*r_tucker + r_tucker^4) * 8;
fprintf('=== Memory Analysis ===\n');
fprintf('Full tensor: %.1f MB\n', full_mem/1024/1024);
fprintf('Tucker format: %.1f KB (%.0fx compression)\n', tucker_mem/1024, full_mem/tucker_mem);
fprintf('\n');

%% Step 6: Test initialize_tensor_lift_tucker_spectral.m (should be equivalent)
fprintf('=======================================================================\n');
fprintf('=== Step 6: Testing initialize_tensor_lift_tucker_spectral.m ===\n');
fprintf('=======================================================================\n');
fprintf('This should produce the SAME results as Spectral Init + RGD above.\n\n');

% Prepare parameters for initialize_tensor_lift_tucker_spectral
params_init = struct();
params_init.T_power = T_rgd;  % Same number of iterations
params_init.mu = mu_rgd;      % Same step size
params_init.r = r_tucker;     % Tucker rank
params_init.Xstar = Xstar;    % Ground truth for error tracking
params_init.verbose = true;   % Show progress
params_init.pre_func = pre_func;  % Same preprocessing

fprintf('Calling initialize_tensor_lift_tucker_spectral with:\n');
fprintf('  T_power: %d, mu: %.4f, r: %d\n', params_init.T_power, params_init.mu, params_init.r);
if use_preprocessing
    fprintf('  Preprocessing: ON\n\n');
else
    fprintf('  Preprocessing: OFF\n\n');
end

tic;
[X_wrapper, ~, history_wrapper] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params_init);
time_wrapper = toc;

% Extract final error from history
error_wrapper_init = history_wrapper.matrix_errors(1);
error_wrapper_final = history_wrapper.matrix_errors(end);
loss_wrapper_final = history_wrapper.loss_function(end);

fprintf('\n=== initialize_tensor_lift_tucker_spectral Results ===\n');
fprintf('Initial error: %.6e\n', error_wrapper_init);
fprintf('Final error:   %.6e (after %d iterations)\n', error_wrapper_final, T_rgd);
fprintf('Final loss:    %.6e\n', loss_wrapper_final);
fprintf('Total time:    %.2f seconds\n\n', time_wrapper);

%% Step 7: Comparison of Two Methods
fprintf('=======================================================================\n');
fprintf('=== Step 7: Comparison of Two Methods ===\n');
fprintf('=======================================================================\n');
fprintf('Method A: Manual Spectral Init + solve_RGD_tucker_kronecker\n');
fprintf('Method B: initialize_tensor_lift_tucker_spectral (wrapper)\n\n');

fprintf('%-50s | Method A      | Method B      | Difference\n', 'Metric');
fprintf('%-50s-|---------------|---------------|-------------\n', repmat('-', 1, 50));
fprintf('%-50s | %.6e | %.6e | %.2e\n', 'Initial Error', error_init, error_wrapper_init, abs(error_init - error_wrapper_init));
fprintf('%-50s | %.6e | %.6e | %.2e\n', 'Final Error', error_final, error_wrapper_final, abs(error_final - error_wrapper_final));
fprintf('%-50s | %.6e | %.6e | %.2e\n', 'Final Loss', output_rgd.Error_function(end), loss_wrapper_final, abs(output_rgd.Error_function(end) - loss_wrapper_final));
fprintf('%-50s | %.2f s       | %.2f s       | %.2f s\n', 'Computation Time', time_init + time_rgd, time_wrapper, abs((time_init + time_rgd) - time_wrapper));

% Check if results match
error_tol = 1e-8;
loss_tol = 1e-8;

fprintf('\n%-50s | Status\n', 'Validation Check');
fprintf('%-50s-|------------------\n', repmat('-', 1, 50));

if abs(error_init - error_wrapper_init) < error_tol
    fprintf('%-50s | ✓ PASS\n', 'Initial errors match');
else
    fprintf('%-50s | ✗ FAIL (diff=%.2e)\n', 'Initial errors match', abs(error_init - error_wrapper_init));
end

if abs(error_final - error_wrapper_final) < error_tol
    fprintf('%-50s | ✓ PASS\n', 'Final errors match');
else
    fprintf('%-50s | ✗ FAIL (diff=%.2e)\n', 'Final errors match', abs(error_final - error_wrapper_final));
end

if abs(output_rgd.Error_function(end) - loss_wrapper_final) < loss_tol
    fprintf('%-50s | ✓ PASS\n', 'Final losses match');
else
    fprintf('%-50s | ✗ FAIL (diff=%.2e)\n', 'Final losses match', abs(output_rgd.Error_function(end) - loss_wrapper_final));
end

% Compare convergence trajectories
max_error_diff = max(abs(output_rgd.Error_Stand - history_wrapper.matrix_errors));
max_loss_diff = max(abs(output_rgd.Error_function - history_wrapper.loss_function));

fprintf('%-50s | ', 'Error trajectories match (max diff)');
if max_error_diff < error_tol
    fprintf('✓ PASS (%.2e)\n', max_error_diff);
else
    fprintf('✗ FAIL (%.2e)\n', max_error_diff);
end

fprintf('%-50s | ', 'Loss trajectories match (max diff)');
if max_loss_diff < loss_tol
    fprintf('✓ PASS (%.2e)\n', max_loss_diff);
else
    fprintf('✗ FAIL (%.2e)\n', max_loss_diff);
end

% Matrix comparison
X_diff = norm(X_final - X_wrapper, 'fro');
fprintf('%-50s | ', 'Final matrices match (Frobenius norm diff)');
if X_diff < error_tol
    fprintf('✓ PASS (%.2e)\n', X_diff);
else
    fprintf('✗ FAIL (%.2e)\n', X_diff);
end

fprintf('\n');

% Overall verdict
all_pass = (abs(error_final - error_wrapper_final) < error_tol) && ...
           (abs(output_rgd.Error_function(end) - loss_wrapper_final) < loss_tol) && ...
           (max_error_diff < error_tol) && ...
           (max_loss_diff < loss_tol);

if all_pass
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('✓✓✓ ALL TESTS PASSED - Both methods produce identical results!\n');
    fprintf('═══════════════════════════════════════════════════════════\n\n');
else
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('✗✗✗ TESTS FAILED - Methods produce different results!\n');
    fprintf('═══════════════════════════════════════════════════════════\n\n');
    
    % Provide debugging information
    fprintf('Debugging Information:\n');
    fprintf('  - Check that solve_RGD_tucker_kronecker is being called correctly\n');
    fprintf('  - Verify that y_spectral conversion is consistent\n');
    fprintf('  - Ensure preprocessing is applied the same way\n');
    fprintf('  - Check random seed and initialization states\n\n');
end

%% Step 8: Enhanced Convergence Visualization with Comparison
fprintf('=== Generating Enhanced Comparison Plots ===\n');

figure('Position', [100, 100, 1800, 900]);

% Plot 1: Loss convergence comparison
subplot(2, 3, 1);
semilogy(output_rgd.Error_function, 'b-', 'LineWidth', 2, 'DisplayName', 'Method A (Manual)');
hold on;
semilogy(history_wrapper.loss_function, 'r--', 'LineWidth', 2, 'DisplayName', 'Method B (Wrapper)');
xlabel('Iteration'); ylabel('Loss'); 
title('Loss Convergence Comparison'); grid on;
legend('Location', 'best');

% Plot 2: Matrix error convergence comparison
subplot(2, 3, 2);
semilogy(output_rgd.Error_Stand, 'b-', 'LineWidth', 2, 'DisplayName', 'Method A (Manual)');
hold on;
semilogy(history_wrapper.matrix_errors, 'r--', 'LineWidth', 2, 'DisplayName', 'Method B (Wrapper)');
yline(error_init, 'k:', 'Spectral Init', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Matrix Error'); 
title('Error Convergence Comparison'); grid on;
legend('Location', 'best');

% Plot 3: Difference in losses
subplot(2, 3, 3);
loss_diff = abs(output_rgd.Error_function - history_wrapper.loss_function);
semilogy(loss_diff, 'k-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Absolute Difference'); 
title('Loss Difference (Method A - Method B)'); grid on;
ylim([max(1e-16, min(loss_diff)/10), max(loss_diff)*10]);

% Plot 4: Difference in errors
subplot(2, 3, 4);
error_diff = abs(output_rgd.Error_Stand - history_wrapper.matrix_errors);
semilogy(error_diff, 'k-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Absolute Difference'); 
title('Error Difference (Method A - Method B)'); grid on;
ylim([max(1e-16, min(error_diff)/10), max(error_diff)*10]);

% Plot 5: Bar chart comparison
subplot(2, 3, 5);
errors_comparison = [error_init, error_final, error_wrapper_final];
bar(errors_comparison);
set(gca, 'XTickLabel', {'Spectral Init', 'Method A Final', 'Method B Final'});
ylabel('Relative Error');
title('Final Error Comparison');
grid on;

% Plot 6: Timing comparison
subplot(2, 3, 6);
times = [time_init, time_rgd, time_wrapper];
bar(times);
set(gca, 'XTickLabel', {'Spectral Init', 'RGD Refine', 'Wrapper Total'});
ylabel('Time (seconds)');
title('Computation Time Comparison');
grid on;

sgtitle(sprintf('Method Comparison: Spectral Init + RGD (%dx%d, r=%d, m=%d, T=%d)', ...
        d1, d2, r, m, T_rgd));

fprintf('Enhanced comparison plots generated.\n\n');

fprintf('✓ Test Complete\n');

%% Helper Functions

function X = extract_matrix_from_tucker_2(T_tucker)
    % EXTRACT_MATRIX_FROM_TUCKER_2 Extract matrix from Tucker tensor (EFFICIENT METHOD)
    % For 4th-order tensor T = X ⊗ X, extract X via core tensor eigendecomposition
    % Supports non-square matrices (d1≠d2)
    
    d1 = T_tucker.dims(1);
    d2 = T_tucker.dims(2);
    r = T_tucker.tucker_ranks(1);
    
    % Extract factor matrices and core
    U1 = T_tucker.U{1};  % d1 × r
    U2 = T_tucker.U{2};  % d2 × r
    U3 = T_tucker.U{3};  % d1 × r
    U4 = T_tucker.U{4};  % d2 × r
    G = T_tucker.G;      % r × r × r × r
    
    % Step 1: Rectify sign ambiguity to enforce U1≈U3, U2≈U4
    U3_rect = U3;
    U4_rect = U4;
    
    for i = 1:r
        % Align U3(:,i) with U1(:,i)
        if dot(U1(:,i), U3(:,i)) < 0
            U3_rect(:,i) = -U3(:,i);
        end
        % Align U4(:,i) with U2(:,i)
        if dot(U2(:,i), U4(:,i)) < 0
            U4_rect(:,i) = -U4(:,i);
        end
    end
    
    % Sign correction matrices
    S3 = diag(sign(diag(U3_rect' * U3)));
    S4 = diag(sign(diag(U4_rect' * U4)));
    
    % Apply sign corrections to core
    G_rect = G;
    for c = 1:r
        for d = 1:r
            G_rect(:, :, c, d) = G(:, :, c, d) * S3(c,c) * S4(d,d);
        end
    end
    
    % Step 2: Form G_mat (r² × r²) from G_rect
    G_mat = reshape(G_rect, [r*r, r*r]);
    
    % Step 3: Eigendecompose G_mat (r² × r²)
    [V_G, D_G] = eig(G_mat);
    [lambda_max, idx_max] = max(abs(diag(D_G)));
    q = V_G(:, idx_max);  % Leading eigenvector (r² × 1)
    
    % Step 4: Reconstruct v = (U2 ⊗ U1) * q
    U_kron = kron(U2, U1);  % (d1*d2) × r²
    v = U_kron * q;         % (d1*d2) × 1
    
    % Scale by eigenvalue
    v = v * sqrt(abs(lambda_max));
    
    % Step 5: Reshape v to get X (d1 × d2)
    X = reshape(v, [d1, d2]);
    
    % Normalize
    X = X / norm(X, 'fro');
end

