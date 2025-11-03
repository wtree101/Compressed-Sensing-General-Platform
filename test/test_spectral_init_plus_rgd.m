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

fprintf('=== Spectral Initialization + RGD Refinement Test ===\n\n');

%% Setup
d = 40; r = 1; r_tucker = r; m = 450; 
mu_rgd = 0.1;   % Step size for RGD refinement
T_rgd = 200;    % Number of RGD refinement iterations
use_preprocessing = false;  % Set to false to disable preprocessing
rng(42);

% Ground truth
U_true = randn(d, r);
Xstar = U_true * U_true';
Xstar = Xstar / norm(Xstar, 'fro');

% Create measurement operator (phase retrieval measurements)
A = randn(m, d*d);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d, d]);

% Generate measurements: phase retrieval (magnitude only)
y = abs(operator.A(Xstar)) / sqrt(m);

% Define preprocessing function for measurements
if use_preprocessing
    pre_func = @(y) set_zero_outside_range_tensor(y);
else
    pre_func = @(y) y;  % Identity function (no preprocessing)
end

fprintf('Setup: d=%d, r=%d, r_tucker=%d, m=%d\n', d, r, r_tucker, m);
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

n = d * d;
A_matrix_extracted = zeros(m, n);
for j = 1:n
    e_j = zeros(n, 1);
    e_j(j) = 1;
    E_j = reshape(e_j, [d, d]);
    A_matrix_extracted(:, j) = operator.A(E_j);
end

% Create cell array of measurement matrices
A_cells = cell(m, 1);
for i = 1:m
    Ai = reshape(A_matrix_extracted(i, :), [d, d]);
    A_cells{i} = (Ai + Ai') / 2;  % Symmetrize
end

% Create Tucker operator
tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false);
tucker_op.A_mat = A_matrix_extracted';  % Store for efficient computation (d² × m)

fprintf('TuckerOperator created with %d measurement matrices.\n\n', m);

%% Step 2: Spectral Initialization using TuckerTensor.initialize_spectral
fprintf('=== Step 2: Spectral Initialization ===\n');

% Create TuckerTensor object for initialization
dims = [d, d, d, d];
T_tucker_init = TuckerTensor(dims, r_tucker, 'symmetric', true, 'init_method', 'zeros');

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
X_init = extract_matrix_from_tucker(T_tucker_init);
X_init = (X_init + X_init') / 2;

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
[output_rgd, T_tucker_final] = solve_RGD_tucker_kronecker(T_tucker_init, [], y_spectral, tucker_op, d, d, r_tucker, m, params_rgd);
time_rgd = toc;

% Extract final matrix
X_final = extract_matrix_from_tucker(T_tucker_final);
X_final = (X_final + X_final') / 2;

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
sgtitle(sprintf('Spectral Init + RGD (d=%d, r=%d, m=%d, T=%d) - Preprocessing: %s', ...
        d, r, m, T_rgd, preproc_str));

fprintf('Convergence plots generated.\n\n');

%% Memory Analysis
full_mem = d^4 * 8;
tucker_mem = (4*d*r_tucker + r_tucker^4) * 8;
fprintf('=== Memory Analysis ===\n');
fprintf('Full tensor: %.1f MB\n', full_mem/1024/1024);
fprintf('Tucker format: %.1f KB (%.0fx compression)\n', tucker_mem/1024, full_mem/tucker_mem);
fprintf('\n');

fprintf('✓ Test Complete\n');

%% Helper Functions

function X = extract_matrix_from_tucker(T_tucker)
    % EXTRACT_MATRIX_FROM_TUCKER Extract matrix from Tucker tensor
    % For 4th-order tensor T = X ⊗ X, extract X via matricization
    
    d = T_tucker.dims(1);
    r = T_tucker.tucker_ranks(1);
    
    U1 = T_tucker.U{1};
    U2 = T_tucker.U{2};
    U3 = T_tucker.U{3};
    U4 = T_tucker.U{4};
    G = T_tucker.G;
    
    % Handle rank-1 case (scalar core)
    if isscalar(G)
        U_left = kron(U1, U2);
        U_right = kron(U3, U4);
        T_mat = G * (U_left * U_right');
    else
        G_mat = reshape(permute(G, [1,2,3,4]), [r*r, r*r]);
        U_left = kron(U1, U2);
        U_right = kron(U3, U4);
        T_mat = U_left * G_mat * U_right';
    end
    
    T_mat = (T_mat + T_mat') / 2;
    [V, D] = eig(T_mat);
    [~, idx] = max(abs(diag(D)));
    v_lead = V(:, idx);
    X = reshape(v_lead, [d, d]);
    X = X / norm(X, 'fro');
    X = (X + X') / 2;
end

