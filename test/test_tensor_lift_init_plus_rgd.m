%% Test Tensor Lift Initialization + RGD Refinement
% Test two-stage approach:
%   1. Initialize using initialize_tensor_lift (T_power steps) - returns matrix
%   2. Convert matrix to TuckerTensor
%   3. Refine using solve_RGD_tucker_kronecker (T steps)

clear; clc;

fprintf('=== Tensor Lift Initialization + RGD Refinement Test ===\n\n');

%% Setup
d = 40; r = 1; r_tucker = r; m = 550; 
mu_init = 1;   % Step size for tensor lift initialization
mu_rgd = 0.1;   % Step size for RGD refinement
T_power = 1;    % Number of initialization iterations
T_rgd = 200;     % Number of RGD refinement iterations
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

%% Step 2: Initialize using initialize_tensor_lift (returns matrix)
fprintf('=== Step 2: Tensor Lift Initialization (%d iterations) ===\n', T_power);

params_init = struct('T_power', T_power, 'mu', mu_init, 'r', r, ...
                    'Xstar', Xstar, 'verbose', 1);

tic;
[X_init, ~, hist_init] = initialize_tensor_lift(y, operator, d, d, params_init);
time_init = toc;

[error_init, ~] = rectify_sign_ambiguity(X_init, Xstar);

fprintf('Initialization complete: Error=%.2e, Time=%.2fs\n', error_init, time_init);
if isfield(hist_init, 'loss_function') && ~isempty(hist_init.loss_function)
    fprintf('Final loss: %.2e\n\n', hist_init.loss_function(end));
else
    fprintf('\n');
end

%% Step 3: Reconstruct TuckerTensor from initialized matrix
% Convert the matrix X_init to TuckerTensor format for RGD refinement
fprintf('=== Step 3: Converting Matrix to TuckerTensor ===\n');

% Symmetrize X_init
X_init_sym = (X_init + X_init') / 2;

% Create TuckerTensor dimensions
dims = [d, d, d, d];

% For rank-r_tucker Tucker tensor, use SVD of X_init to get factors
[U_svd, S_svd, ~] = svd(X_init_sym);
r_effective = r_tucker;

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

% Create core tensor: compute by contracting full tensor with U'
if r_tucker == 1
    % For rank-1, core is scalar representing the scale
    if r_effective >= 1
        % Use first singular value
        T_tucker_init.G = sqrt(S_factor(1,1));
    else
        T_tucker_init.G = norm(X_init_sym, 'fro') / d;
    end
else
    % For higher rank, use HOSVD to convert full tensor to Tucker format
    % Create tensor T = X_init ⊗ X_init
    T_full_temp = create_tensor_from_matrix(X_init_sym, d);
    
    % Use HOSVD to decompose into Tucker format
    % HOSVD returns tensor with projected factors, need to extract core
    T_hosvd = HOSVD(T_full_temp, r_tucker*ones(1, 4));
    
    % Extract core by contracting with U' (factors from HOSVD may differ)
    % So we compute core directly: G = T ×₁ U₁' ×₂ U₂' ×₃ U₃' ×₄ U₄'
    G_init = T_full_temp;
    for mode = 1:4
        G_init = tensor_mode_product(G_init, T_tucker_init.U{mode}', mode);
    end
    % Core should be size (r_tucker × r_tucker × r_tucker × r_tucker)
    T_tucker_init.G = G_init;
end

fprintf('TuckerTensor created: rank=%d\n', r_tucker);
if isscalar(T_tucker_init.G)
    fprintf('Core (scalar): %.6f\n', T_tucker_init.G);
else
    fprintf('Core norm: %.6f\n', norm(T_tucker_init.G(:)));
end
fprintf('\n');

%% Step 4: RGD Refinement
fprintf('=== Step 4: RGD Refinement (%d iterations) ===\n', T_rgd);

% Convert phase retrieval measurements to tensor measurements
% For T = X ⊗ X, we have: <A_i ⊗ A_i, X ⊗ X> = (<A_i, X>)^2
% Original measurement: y_i = |<A_i, X>| / sqrt(m)
% So (<A_i, X>)^2 = (y_i * sqrt(m))^2 = y_i^2 * m
% solve_RGD_tucker_kronecker computes: y_pred = forward(T) / sqrt(m) = (<A_i, X>)^2 / sqrt(m)
% To match, we need: y_tensor = (<A_i, X>)^2 / sqrt(m) = (y_i^2 * m) / sqrt(m) = y.^2 * sqrt(m)
y_tensor = y.^2 * sqrt(m);

params_rgd = struct('T', T_rgd, 'mu', mu_rgd, 'Xstar', Xstar, 'verbose', 1);

tic;
[output_rgd, T_tucker_final] = solve_RGD_tucker_kronecker(T_tucker_init, [], y_tensor, tucker_op, d, d, r_tucker, m, params_rgd);
time_rgd = toc;

% Extract final matrix
X_final = extract_matrix_from_tucker(T_tucker_final);
X_final = (X_final + X_final') / 2;

[error_final, ~] = rectify_sign_ambiguity(X_final, Xstar);

fprintf('RGD refinement complete: Error=%.2e, Time=%.2fs\n', error_final, time_rgd);
fprintf('Final loss: %.2e\n\n', output_rgd.Error_function(end));

%% Step 5: Results Summary
fprintf('=== Results Summary ===\n');
fprintf('Tensor Lift Initialization:\n');
fprintf('  Error: %.6e (after %d iterations)\n', error_init, T_power);
if isfield(hist_init, 'loss_function') && ~isempty(hist_init.loss_function)
    fprintf('  Loss:  %.6e\n', hist_init.loss_function(end));
end
fprintf('  Time:  %.2f seconds\n', time_init);

fprintf('\nRGD Refinement:\n');
fprintf('  Error: %.6e (after %d iterations)\n', error_final, T_rgd);
fprintf('  Loss:  %.6e\n', output_rgd.Error_function(end));
fprintf('  Time:  %.2f seconds\n', time_rgd);

fprintf('\nOverall Improvement:\n');
improvement = (error_init - error_final) / error_init * 100;
fprintf('  Error reduction: %.2f%%\n', improvement);
if isfield(hist_init, 'loss_function') && ~isempty(hist_init.loss_function)
    fprintf('  Loss reduction:  %.2f%%\n', ...
            (hist_init.loss_function(end) - output_rgd.Error_function(end)) / ...
            hist_init.loss_function(end) * 100);
end

fprintf('\nTotal time: %.2f seconds\n\n', time_init + time_rgd);

%% Step 6: Convergence Visualization
fprintf('=== Generating Convergence Plots ===\n');

figure('Position', [100, 100, 1400, 500]);

% Plot 1: Loss convergence (both stages)
subplot(1, 4, 1);
if isfield(hist_init, 'loss_function') && ~isempty(hist_init.loss_function)
    iter_init = 1:length(hist_init.loss_function);
    iter_rgd = (length(hist_init.loss_function)+1):(length(hist_init.loss_function)+length(output_rgd.Error_function));
    semilogy(iter_init, hist_init.loss_function, 'b-', 'LineWidth', 2); hold on;
    semilogy(iter_rgd, output_rgd.Error_function, 'r-', 'LineWidth', 2);
    xline(length(hist_init.loss_function), 'k--', 'Init→RGD', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Loss'); 
    title('Loss: Tensor Lift (blue) + RGD (red)'); grid on; legend('Tensor Lift', 'RGD', 'Location', 'best');
else
    % Only RGD losses available
    semilogy(output_rgd.Error_function, 'r-', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Loss'); 
    title('RGD Loss Convergence'); grid on;
end

% Plot 2: Matrix error convergence (both stages)
subplot(1, 4, 2);
if isfield(hist_init, 'tensor_errors') && ~isempty(hist_init.tensor_errors)
    % Use tensor errors as proxy for initialization errors
    iter_init_err = 1:length(hist_init.tensor_errors);
    iter_rgd_err = (length(hist_init.tensor_errors)+1):(length(hist_init.tensor_errors)+length(output_rgd.Error_Stand));
    semilogy(iter_init_err, hist_init.tensor_errors / norm(Xstar, 'fro'), 'b-', 'LineWidth', 2); hold on;
    semilogy(iter_rgd_err, output_rgd.Error_Stand, 'r-', 'LineWidth', 2);
    xline(length(hist_init.tensor_errors), 'k--', 'Init→RGD', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Matrix Error'); 
    title('Error: Tensor Lift (blue) + RGD (red)'); grid on; legend('Tensor Lift', 'RGD', 'Location', 'best');
else
    % Only RGD errors available
    semilogy(output_rgd.Error_Stand, 'r-', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Matrix Error'); 
    title('RGD Error Convergence'); grid on;
end

% Plot 3: Initialization convergence detail
subplot(1, 4, 3);
if isfield(hist_init, 'loss_function') && ~isempty(hist_init.loss_function)
    semilogy(hist_init.loss_function, 'b-', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Loss'); 
    title(sprintf('Tensor Lift Init (T=%d)', T_power)); grid on;
else
    text(0.5, 0.5, 'Loss data unavailable', 'HorizontalAlignment', 'center');
    axis off;
end

% Plot 4: RGD convergence detail
subplot(1, 4, 4);
semilogy(output_rgd.Error_function, 'r-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Loss'); 
title(sprintf('RGD Refinement (T=%d)', T_rgd)); grid on;

sgtitle(sprintf('Two-Stage Convergence: Tensor Lift (T=%d) + RGD (T=%d)', T_power, T_rgd));

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

