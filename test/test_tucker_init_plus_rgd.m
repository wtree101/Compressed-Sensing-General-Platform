%% Test Tucker Tensor Initialization + RGD Refinement
% Test two-stage approach:
%   1. Initialize using initialize_tensor_lift_tucker (T_power steps)
%   2. Refine using solve_RGD_tucker_kronecker (T steps)

clear; clc;

fprintf('=== Tucker Tensor Initialization + RGD Refinement Test ===\n\n');

%% Setup
d = 20; r = 1; r_tucker = r; m = 400; 
mu_init = 0.005;  % Step size for initialization
mu_rgd = 0.01;    % Step size for RGD refinement
T_power = 100;    % Number of initialization iterations
T_rgd = 200;      % Number of RGD refinement iterations
rng(42);

% Ground truth
U_true = randn(d, r);
Xstar = U_true * U_true';
Xstar = Xstar / norm(Xstar, 'fro');

% Measurements: y_i = <A_i ⊗ A_i, X ⊗ X>
A_matrix = randn(m, d*d);
y = zeros(m, 1);
for i = 1:m
    Ai = reshape(A_matrix(i, :), [d, d]);
    Ai = (Ai + Ai') / 2;
    y(i) = trace(Ai * Xstar * Ai * Xstar);
end
y = y / sqrt(m);

operator = struct();
operator.A = @(X) forward_op(X, A_matrix, d);

fprintf('Setup: d=%d, r=%d, r_tucker=%d, m=%d\n', d, r, r_tucker, m);
fprintf('Init iterations: %d, RGD iterations: %d\n\n', T_power, T_rgd);

%% Step 1: Extract measurement matrices and create TuckerOperator
% (Same as in initialize_tensor_lift_tucker.m)
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

fprintf('TuckerOperator created.\n\n');

%% Step 2: Initialize using initialize_tensor_lift_tucker
fprintf('=== Step 2: Tucker Tensor Initialization (%d iterations) ===\n', T_power);

params_init = struct('T_power', T_power, 'mu', mu_init, 'r', r_tucker, ...
                    'Xstar', Xstar, 'verbose', 1, 'symmetric', true, ...
                    'init_method', 'ones');

tic;
[X_init, ~, hist_init] = initialize_tensor_lift_tucker(y, operator, d, d, params_init);
time_init = toc;

[error_init, ~] = rectify_sign_ambiguity(X_init, Xstar);

fprintf('Initialization complete: Error=%.2e, Time=%.2fs\n', error_init, time_init);
fprintf('Final loss: %.2e\n\n', hist_init.loss_function(end));

%% Step 3: Reconstruct TuckerTensor from initialized matrix
% We need to create a TuckerTensor from the initialized matrix X_init
% for RGD refinement. For rank-1 case, we can directly create TuckerTensor.

fprintf('=== Step 3: Reconstructing TuckerTensor from Initialized Matrix ===\n');

% Create initial TuckerTensor from X_init
% Method: For rank-r case, use SVD of X_init to get factors
dims = [d, d, d, d];

% Symmetrize X_init
X_init_sym = (X_init + X_init') / 2;

% For rank-1 Tucker, we can directly construct from X_init's leading eigenvector
[U_svd, S_svd, ~] = svd(X_init_sym);
r_effective = min(r_tucker, rank(X_init_sym, 1e-10));

% Use leading r_effective singular vectors
U_factor = U_svd(:, 1:r_effective);
S_factor = S_svd(1:r_effective, 1:r_effective);

% Create TuckerTensor with symmetric structure
T_tucker_init = TuckerTensor(dims, r_tucker, 'symmetric', true, 'init_method', 'zeros');

% Set factor matrices (all same for symmetric)
for k = 1:4
    if r_effective == r_tucker
        T_tucker_init.U{k} = U_factor;
    else
        % Pad with random orthogonal vectors if needed
        U_padded = [U_factor, randn(d, r_tucker - r_effective)];
        [U_padded, ~] = qr(U_padded, 0);
        T_tucker_init.U{k} = U_padded(:, 1:r_tucker);
    end
end

% Create core tensor: for rank-1, G is scalar representing the scale
if r_tucker == 1
    % For rank-1, core is scalar = sqrt(S(1,1)) or use trace approximation
    if r_effective >= 1
        T_tucker_init.G = sqrt(S_factor(1,1));
    else
        T_tucker_init.G = norm(X_init_sym, 'fro') / d;
    end
else
    % For higher rank, need to compute core by contracting with factors
    % Create tensor T = X_init ⊗ X_init and contract with U'
    T_full_temp = create_tensor_from_matrix(X_init_sym, d);
    G_init = T_full_temp;
    for mode = 1:4
        G_init = tensor_mode_product(G_init, T_tucker_init.U{mode}', mode);
    end
    % Truncate to r_tucker
    G_init_perm = permute(G_init, [1,2,3,4]);
    G_init_vec = G_init_perm(:);
    G_init_mat = reshape(G_init_vec, [r_tucker^2, r_tucker^2]);
    [UG, SG, ~] = svd(G_init_mat);
    G_truncated = UG(:, 1:r_tucker) * sqrt(SG(1:r_tucker, 1:r_tucker)) * UG(:, 1:r_tucker)';
    T_tucker_init.G = reshape(G_truncated, [r_tucker, r_tucker, r_tucker, r_tucker]);
end

fprintf('TuckerTensor reconstructed: rank=%d\n', r_tucker);
if isscalar(T_tucker_init.G)
    fprintf('Core (scalar): %.6f\n', T_tucker_init.G);
else
    fprintf('Core norm: %.6f\n', norm(T_tucker_init.G(:)));
end
fprintf('\n');

%% Step 4: RGD Refinement
fprintf('=== Step 4: RGD Refinement (%d iterations) ===\n', T_rgd);

params_rgd = struct('T', T_rgd, 'mu', mu_rgd, 'Xstar', Xstar, 'verbose', 1);

tic;
[output_rgd, T_tucker_final] = solve_RGD_tucker_kronecker(T_tucker_init, [], y, tucker_op, d, d, r_tucker, m, params_rgd);
time_rgd = toc;

% Extract final matrix
X_final = extract_matrix_from_tucker(T_tucker_final);
X_final = (X_final + X_final') / 2;

[error_final, ~] = rectify_sign_ambiguity(X_final, Xstar);

fprintf('RGD refinement complete: Error=%.2e, Time=%.2fs\n', error_final, time_rgd);
fprintf('Final loss: %.2e\n\n', output_rgd.Error_function(end));

%% Step 5: Results Summary
fprintf('=== Results Summary ===\n');
fprintf('Initialization:\n');
fprintf('  Error: %.6e (after %d iterations)\n', error_init, T_power);
fprintf('  Loss:  %.6e\n', hist_init.loss_function(end));
fprintf('  Time:  %.2f seconds\n', time_init);

fprintf('\nRGD Refinement:\n');
fprintf('  Error: %.6e (after %d iterations)\n', error_final, T_rgd);
fprintf('  Loss:  %.6e\n', output_rgd.Error_function(end));
fprintf('  Time:  %.2f seconds\n', time_rgd);

fprintf('\nOverall Improvement:\n');
improvement = (error_init - error_final) / error_init * 100;
fprintf('  Error reduction: %.2f%%\n', improvement);
fprintf('  Loss reduction:  %.2f%%\n', ...
        (hist_init.loss_function(end) - output_rgd.Error_function(end)) / ...
        hist_init.loss_function(end) * 100);

fprintf('\nTotal time: %.2f seconds\n\n', time_init + time_rgd);

%% Step 6: Convergence Visualization
fprintf('=== Generating Convergence Plots ===\n');

figure('Position', [100, 100, 1400, 500]);

% Plot 1: Loss convergence (both stages)
subplot(1, 4, 1);
iter_init = 1:length(hist_init.loss_function);
iter_rgd = (length(hist_init.loss_function)+1):(length(hist_init.loss_function)+length(output_rgd.Error_function));
semilogy(iter_init, hist_init.loss_function, 'b-', 'LineWidth', 2); hold on;
semilogy(iter_rgd, output_rgd.Error_function, 'r-', 'LineWidth', 2);
xline(length(hist_init.loss_function), 'k--', 'Init→RGD', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Loss'); 
title('Loss: Init (blue) + RGD (red)'); grid on; legend('Init', 'RGD', 'Location', 'best');

% Plot 2: Matrix error convergence (both stages)
subplot(1, 4, 2);
if isfield(hist_init, 'matrix_errors') && ~isempty(hist_init.matrix_errors)
    iter_init_err = 1:length(hist_init.matrix_errors);
    iter_rgd_err = (length(hist_init.matrix_errors)+1):(length(hist_init.matrix_errors)+length(output_rgd.Error_Stand));
    semilogy(iter_init_err, hist_init.matrix_errors, 'b-', 'LineWidth', 2); hold on;
    semilogy(iter_rgd_err, output_rgd.Error_Stand, 'r-', 'LineWidth', 2);
    xline(length(hist_init.matrix_errors), 'k--', 'Init→RGD', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Matrix Error'); 
    title('Error: Init (blue) + RGD (red)'); grid on; legend('Init', 'RGD', 'Location', 'best');
else
    % Only RGD errors available
    semilogy(output_rgd.Error_Stand, 'r-', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Matrix Error'); 
    title('RGD Error Convergence'); grid on;
end

% Plot 3: Initialization convergence detail
subplot(1, 4, 3);
semilogy(hist_init.loss_function, 'b-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Loss'); 
title(sprintf('Initialization (T=%d)', T_power)); grid on;

% Plot 4: RGD convergence detail
subplot(1, 4, 4);
semilogy(output_rgd.Error_function, 'r-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Loss'); 
title(sprintf('RGD Refinement (T=%d)', T_rgd)); grid on;

sgtitle(sprintf('Two-Stage Convergence: Init (T=%d) + RGD (T=%d)', T_power, T_rgd));

fprintf('Convergence plots generated.\n\n');

%% Memory Analysis
full_mem = d^4 * 8;
tucker_mem = (4*d*r_tucker + r_tucker^4) * 8;
fprintf('=== Memory Analysis ===\n');
fprintf('Full tensor: %.1f MB\n', full_mem/1024/1024);
fprintf('Tucker format: %.1f KB (%.0fx compression)\n', tucker_mem/1024, full_mem/tucker_mem);
fprintf('\n');

fprintf('✓ Test Complete\n');

%% Helper function
function y = forward_op(X, A_matrix, d)
    % Forward operator: y = A * vec(X ⊗ X)
    m = size(A_matrix, 1);
    y = zeros(m, 1);
    for i = 1:m
        Ai = reshape(A_matrix(i, :), [d, d]);
        Ai = (Ai + Ai') / 2;
        y(i) = trace(Ai * X * Ai * X);
    end
end

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

function T_out = tensor_mode_product(T, M, mode)
    % TENSOR_MODE_PRODUCT n-mode product of tensor T with matrix M
    % T_out = T ×_mode M
    
    sz = size(T);
    k = size(M, 1);
    
    % Permute so mode is first
    order = 1:max(ndims(T), mode);
    order([1, mode]) = [mode, 1];
    T_perm = permute(T, order);
    
    % Reshape to matrix and multiply
    T_mat = reshape(T_perm, sz(mode), []);
    T_out_mat = M * T_mat;
    
    % Reshape back
    sz_out = sz;
    sz_out(mode) = k;
    sz_out_perm = sz_out(order);
    T_out_perm = reshape(T_out_mat, sz_out_perm);
    
    % Permute back
    T_out = ipermute(T_out_perm, order);
end

