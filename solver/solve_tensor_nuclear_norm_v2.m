function [X_recovered, info] = solve_tensor_nuclear_norm_v2(operator, y, d, varargin)
% SOLVE_TENSOR_NUCLEAR_NORM_V2 Tensor nuclear norm minimization for phase retrieval
%
% This solver recovers a symmetric matrix X from measurements y using
% tensor nuclear norm minimization on the lifted tensor T = X ⊗ X.
%
% Problem formulation:
%   min_T  sum_{k=1}^4 lambda_k ||T_(k)||_*
%   s.t.   y_i = <A_i, T> / sqrt(m), for all i
%
% where:
%   A_i = A_i ⊗ A_i (4th-order measurement tensors)
%   T = X ⊗ X (4th-order target tensor)
%   T_(k) is the mode-k unfolding (matricization) of tensor T
%   ||·||_* is the nuclear norm (sum of singular values)
%   lambda_k are weights for each mode (default: [1,1,1,1])
%
% ADMM Algorithm:
%   Auxiliary variables: W_k = T_(k), k=1,...,4
%   Dual variables: Z_k for W_k = T_(k), u_i for measurement constraints
%   Penalty parameters: rho_k for unfolding constraints, rho_m for measurements
%
% Inputs:
%   operator - Struct with measurement operators:
%              .A_cells: Cell array {A_1, ..., A_m} of d×d matrices
%   y        - Measurement vector (m × 1)
%   d        - Dimension of the matrix X (X is d × d)
%
% Optional Name-Value Pairs:
%   'max_iter'    - Maximum number of iterations (default: 1000)
%   'tol'         - Convergence tolerance (default: 1e-6)
%   'lambda'      - Weight vector [lambda_1,...,lambda_4] for mode nuclear norms (default: [1,1,1,1])
%   'rho_k'       - Penalty parameter for unfolding constraints (default: 0.1)
%   'rho_m'       - Penalty parameter for measurement constraints (default: 1.0)
%   'verbose'     - Verbosity level: 0=silent, 1=basic, 2=detailed (default: 1)
%   'X_true'      - Ground truth for computing error (optional)
%   'rank'        - Target rank for matrix extraction (default: rank of X_true or 5)
%   'pcg_tol'     - PCG tolerance for T-update (default: 1e-3)
%   'pcg_maxit'   - PCG max iterations (default: 100)
%
% Outputs:
%   X_recovered - Recovered d×d matrix
%   info        - Struct with solver information:
%                 .obj_values: Objective values per iteration
%                 .errors: Reconstruction errors (if X_true provided)
%                 .times: Computation time per iteration
%                 .primal_residuals: Primal residuals per iteration
%                 .measurement_residuals: Measurement constraint violations
%                 .iter: Total iterations
%                 .converged: Convergence flag

% Parse inputs
p = inputParser;
addRequired(p, 'operator');
addRequired(p, 'y');
addRequired(p, 'd');
addParameter(p, 'max_iter', 1000, @isnumeric);
addParameter(p, 'tol', 1e-6, @isnumeric);
addParameter(p, 'lambda', [1, 1, 1, 1], @isnumeric);  % Weights for mode nuclear norms
addParameter(p, 'rho_k', 1.0, @isnumeric);            % Penalty for unfolding constraints
addParameter(p, 'rho_m', 1.0, @isnumeric);            % Penalty for measurement constraints
addParameter(p, 'verbose', 1, @isnumeric);
addParameter(p, 'X_true', [], @isnumeric);
addParameter(p, 'rank', [], @isnumeric);
addParameter(p, 'pcg_tol', 1e-10, @isnumeric);
addParameter(p, 'pcg_maxit', 1000, @isnumeric);
addParameter(p, 'use_spectral_init', true, @islogical);

parse(p, operator, y, d, varargin{:});

max_iter = p.Results.max_iter;
tol = p.Results.tol;
lambda_vec = p.Results.lambda(:)';  % [lambda_1, ..., lambda_4]
rho_k = p.Results.rho_k;
rho_m = p.Results.rho_m;
verbose = p.Results.verbose;
X_true = p.Results.X_true;
target_rank = p.Results.rank;
pcg_tol = p.Results.pcg_tol;
pcg_maxit = p.Results.pcg_maxit;
use_spectral_init = p.Results.use_spectral_init;

% Validate inputs
% if ~isfield(operator, 'A_cells')
%     error('operator must have A_cells field');
% end
m = length(y);
if length(operator.A_cells) ~= m
    error('Number of measurement matrices must equal length of y');
end
if length(lambda_vec) ~= 4
    error('lambda must be a vector of length 4');
end

% Print header
if verbose >= 1
    fprintf('\n=== Tensor Nuclear Norm Minimization (v2) ===\n');
    fprintf('Problem size: d=%d, measurements m=%d\n', d, m);
    fprintf('Tensor size: %d×%d×%d×%d\n', d, d, d, d);
    fprintf('Parameters: rho_k=%.2e, rho_m=%.2e\n', rho_k, rho_m);
    fprintf('Nuclear norm weights: [%.2f, %.2f, %.2f, %.2f]\n', lambda_vec);
    fprintf('Max iterations: %d, tolerance: %.2e\n', max_iter, tol);
    fprintf('PCG: tol=%.0e, maxit=%d\n', pcg_tol, pcg_maxit);
    fprintf('\n');
end

% Compute ground truth tensor metrics if X_true provided
if ~isempty(X_true) && verbose >= 1
    T_true = create_tensor_from_matrix(X_true, d);
    fprintf('Ground truth tensor metrics:\n');
    fprintf('  ||T_true||_F = %.6f\n', norm(T_true(:)));
    
    % Compute objective value at T_true
    obj_val_true = 0;
    for k = 1:4
        obj_val_true = obj_val_true + lambda_vec(k) * sum(svd(tensor_mode_unfold(T_true, k)));
    end
    fprintf('  Objective at T_true: %.6e\n', obj_val_true);
    
    % Compute measurement residual at T_true
    meas_res_true = 0;
    for i = 1:m
        Ai_tensor = compute_Ai_tensor(operator.A_cells{i});
        meas_res_true = meas_res_true + (y(i) - tensor_inner_product(Ai_tensor, T_true) / sqrt(m))^2;
    end
    meas_res_true = sqrt(meas_res_true);
    fprintf('  Measurement residual at T_true: %.6e\n', meas_res_true);
    
    % Compute mode unfolding metrics for T_true
    fprintf('  Mode unfolding properties:\n');
    for k = 1:4
        T_true_k = tensor_mode_unfold(T_true, k);
        norm_k = norm(T_true_k, 'fro');
        rank_k = rank(T_true_k, 1e-6);
        nuclear_k = sum(svd(T_true_k));
        fprintf('    Mode %d: ||T_(%d)||_F = %.6f, rank = %d, ||T_(%d)||_* = %.6f\n', ...
            k, k, norm_k, rank_k, k, nuclear_k);
    end
    fprintf('\n');
    
    % Store for later comparison
    info.obj_val_true = obj_val_true;
    info.meas_res_true = meas_res_true;
else
    T_true = [];
end

% Initialize
tensor_dims = [d, d, d, d];

% Warm start initialization
if use_spectral_init
    % Kronecker adjoint: T0 = (1/√m) Σ_i y_i (A_i ⊗ A_i)
    T = zeros(tensor_dims);
    inv_sqrt_m = 1 / sqrt(m);
    for i = 1:m
        if y(i) ~= 0
            Ai_tensor = compute_Ai_tensor(operator.A_cells{i});
            T = T + (inv_sqrt_m * y(i)) * Ai_tensor;
        end
    end
    if verbose >= 1
        fprintf('Spectral initialization: ||T0|| = %.6e\n', norm(T(:)));
    end
else
    T = zeros(tensor_dims);
end

% Initialize info structure
info = struct();
info.obj_values = zeros(max_iter, 1);
info.primal_residuals = zeros(max_iter, 1);
info.measurement_residuals = zeros(max_iter, 1);
info.times = zeros(max_iter, 1);
info.converged = false;
if ~isempty(X_true)
    info.errors = zeros(max_iter, 1);
end

% ADMM variables
W = cell(1, 4);      % Auxiliary variables W_k for each mode unfolding
Z = cell(1, 4);      % Dual variables Z_k (scaled) for W_k = T_(k)
u = zeros(m, 1);     % Dual variables u_i (scaled) for measurement constraints

% Initialize W_k and Z_k for each mode
unfolding_sizes = compute_unfolding_sizes(d);
for k = 1:4
    W{k} = tensor_mode_unfold(T, k);
    Z{k} = zeros(unfolding_sizes{k});
end

start_time = tic;

%% Main ADMM iteration
for iter = 1:max_iter
    iter_start = tic;
    
    % Step 1: Update W_k (Singular Value Thresholding)
    % W_k = SVT_{lambda_k/rho_k}(T_(k) - Z_k)
    for k = 1:4
        T_k = tensor_mode_unfold(T, k);
        tau = lambda_vec(k) / rho_k;
        W{k} = singular_value_thresholding(T_k - Z{k}, tau);
    end
    
    % Step 2: Update T (Least Squares via Conjugate Gradient)
    % min_T (rho_k/2) * sum_k ||W_k - T_(k) + Z_k||_F^2
    %       + (rho_m/2) * sum_i (y_i - <A_i, T>/sqrt(m) + u_i)^2
    T = update_T_least_squares_v2(T, W, Z, operator.A_cells, y, u, rho_k, rho_m, ...
                                   tensor_dims, m, pcg_tol, pcg_maxit);
    
    % Step 3: Update dual variables Z_k
    % Z_k = Z_k + W_k - T_(k)
    for k = 1:4
        T_k = tensor_mode_unfold(T, k);
        Z{k} = Z{k} + W{k} - T_k;
    end
    
    % Step 4: Update dual variables u_i
    % u_i = u_i + y_i - <A_i, T>/sqrt(m)
    for i = 1:m
        Ai_tensor = compute_Ai_tensor(operator.A_cells{i});
        measurement = tensor_inner_product(Ai_tensor, T) / sqrt(m);
        u(i) = u(i) + y(i) - measurement;
    end
    
    % Compute objective value
    obj_val = 0;
    for k = 1:4
        obj_val = obj_val + lambda_vec(k) * sum(svd(tensor_mode_unfold(T, k)));
    end
    
    % Compute primal residual (unfolding constraints)
    primal_res = 0;
    for k = 1:4
        T_k = tensor_mode_unfold(T, k);
        primal_res = primal_res + norm(W{k} - T_k, 'fro')^2;
    end
    primal_res = sqrt(primal_res);
    
    % Compute measurement residual
    meas_res = 0;
    for i = 1:m
        Ai_tensor = compute_Ai_tensor(operator.A_cells{i});
        meas_res = meas_res + (y(i) - tensor_inner_product(Ai_tensor, T) / sqrt(m))^2;
    end
    meas_res = sqrt(meas_res);
    
    % Store iteration info
    info.obj_values(iter) = obj_val;
    info.primal_residuals(iter) = primal_res;
    info.measurement_residuals(iter) = meas_res;
    info.times(iter) = toc(iter_start);
    
    % Compute reconstruction error if X_true provided
    if ~isempty(X_true)
        % Determine rank for extraction
        if isempty(target_rank)
            r_extract = rank(X_true, 1e-6);
        else
            r_extract = target_rank;
        end
        params_extract = struct('r', r_extract, 'method', 'eig', 'verbose', false);
        X_current = extract_matrix_from_tensor(T, params_extract);
        [error_val, ~] = rectify_sign_ambiguity(X_current, X_true);
        info.errors(iter) = error_val;
    end
    
    % Check convergence
    if iter > 1
        obj_change = abs(info.obj_values(iter) - info.obj_values(iter-1)) / ...
                     (abs(info.obj_values(iter-1)) + 1e-10);
        
        converged = (obj_change < tol) && (primal_res < tol) && (meas_res < tol);
        
        if verbose >= 2 || (verbose >= 1 && mod(iter, 10) == 0)
            fprintf('Iter %4d: obj=%.6e, |r_p|=%.2e, |r_m|=%.2e', ...
                iter, obj_val, primal_res, meas_res);
            if ~isempty(X_true)
                fprintf(', err_X=%.6e', info.errors(iter));
                % Compute tensor error if T_true available
                if exist('T_true', 'var') && ~isempty(T_true)
                    tensor_err = norm(T(:) - T_true(:)) / norm(T_true(:));
                    fprintf(', err_T=%.6e', tensor_err);
                end
            end
            fprintf(', ||T||=%.2e\n', norm(T(:)));
        end
        
        if converged
            if verbose >= 1
                fprintf('Converged at iteration %d\n', iter);
            end
            info.converged = true;
            break;
        end
    else
        if verbose >= 2
            fprintf('Iter %4d: obj=%.6e, |r_p|=%.2e, |r_m|=%.2e', ...
                iter, obj_val, primal_res, meas_res);
            if ~isempty(X_true)
                fprintf(', err_X=%.6e', info.errors(iter));
                % Compute tensor error if T_true available
                if exist('T_true', 'var') && ~isempty(T_true)
                    tensor_err = norm(T(:) - T_true(:)) / norm(T_true(:));
                    fprintf(', err_T=%.6e', tensor_err);
                end
            end
            fprintf(', ||T||=%.2e\n', norm(T(:)));
        end
    end
end

total_time = toc(start_time);

% Trim unused entries
info.obj_values = info.obj_values(1:iter);
info.primal_residuals = info.primal_residuals(1:iter);
info.measurement_residuals = info.measurement_residuals(1:iter);
info.times = info.times(1:iter);
if ~isempty(X_true)
    info.errors = info.errors(1:iter);
end
info.iter = iter;
info.total_time = total_time;

% Extract matrix from tensor
if isempty(target_rank)
    if ~isempty(X_true)
        r_extract = rank(X_true, 1e-6);
    else
        r_extract = min(5, d);
    end
else
    r_extract = target_rank;
end
params_extract = struct('r', r_extract, 'method', 'eig', 'verbose', (verbose >= 2));
X_recovered = extract_matrix_from_tensor(T, params_extract);
info.T = T;
% Final summary
if verbose >= 1
    fprintf('\n=== Solver Summary ===\n');
    fprintf('Total iterations: %d\n', iter);
    fprintf('Total time: %.4f seconds\n', total_time);
    fprintf('Converged: %s\n', mat2str(info.converged));
    
    fprintf('\nObjective and residuals:\n');
    fprintf('  Final objective: %.6e\n', info.obj_values(end));
    fprintf('  Final primal residual: %.6e\n', info.primal_residuals(end));
    fprintf('  Final measurement residual: %.6e\n', info.measurement_residuals(end));
    
    if ~isempty(X_true) && isfield(info, 'obj_val_true')
        fprintf('\nComparison with T_true:\n');
        fprintf('  Objective at T_true: %.6e\n', info.obj_val_true);
        fprintf('  Objective at T_recovered: %.6e\n', info.obj_values(end));
        fprintf('  Objective difference: %.6e (%.2f%%)\n', ...
            abs(info.obj_values(end) - info.obj_val_true), ...
            abs(info.obj_values(end) - info.obj_val_true) / info.obj_val_true * 100);
        fprintf('  Measurement residual at T_true: %.6e\n', info.meas_res_true);
        fprintf('  Measurement residual at T_recovered: %.6e\n', info.measurement_residuals(end));
    end
    
    if ~isempty(X_true)
        fprintf('\nReconstruction errors:\n');
        fprintf('  Matrix error (X): %.6e\n', info.errors(end));
        
        % Compute final tensor error
        if exist('T_true', 'var') && ~isempty(T_true)
            tensor_err_final = norm(T(:) - T_true(:)) / norm(T_true(:));
            fprintf('  Tensor error (T): %.6e\n', tensor_err_final);
            fprintf('  Error ratio (T/X): %.4f\n', tensor_err_final / info.errors(end));
            
            % Mode unfolding errors
            fprintf('\nMode unfolding errors:\n');
            for k = 1:4
                T_k = tensor_mode_unfold(T, k);
                T_true_k = tensor_mode_unfold(T_true, k);
                unfold_err = norm(T_k - T_true_k, 'fro') / norm(T_true_k, 'fro');
                fprintf('  Mode %d: %.6e\n', k, unfold_err);
            end
        end
    end
    
    fprintf('\nRecovered properties:\n');
    fprintf('  Matrix rank: %d\n', rank(X_recovered, 1e-6));
    fprintf('  Tensor norm: %.6f\n', norm(T(:)));
    if ~isempty(X_true) && exist('T_true', 'var') && ~isempty(T_true)
        fprintf('  True tensor norm: %.6f\n', norm(T_true(:)));
    end
    fprintf('======================\n\n');
end

end

%% Helper Functions

function T_new = update_T_least_squares_v2(T, W, Z, A_cells, y, u, rho_k, rho_m, ...
                                           tensor_dims, m, pcg_tol, pcg_maxit)
    % Update T by solving least squares problem:
    % min_T (rho_k/2) * sum_k ||W_k - T_(k) + Z_k||_F^2
    %       + (rho_m/2) * sum_i (y_i - <A_i, T>/sqrt(m) + u_i)^2
    %
    % Normal equation:
    % (rho_m/m * A^T A + sum_k rho_k * P_k^T P_k) vec(T) = rhs
    
    T_vec = T(:);
    
    % Define linear operator
    A_op = @(x) apply_T_operator_v2(x, A_cells, rho_k, rho_m, tensor_dims, m);
    
    % Compute right-hand side
    b = compute_rhs_v2(W, Z, A_cells, y, u, rho_k, rho_m, tensor_dims, m);
    
    % Solve with PCG
    [T_vec_new, ~] = pcg(A_op, b, pcg_tol, pcg_maxit, [], [], T_vec);
    
    T_new = reshape(T_vec_new, tensor_dims);
end

function y_out = apply_T_operator_v2(x, A_cells, rho_k, rho_m, tensor_dims, m)
    % Apply the linear operator for T update
    % (rho_m/m * A^T A + sum_k rho_k * P_k^T P_k) x
    
    T = reshape(x, tensor_dims);
    
    % Term 1: sum_k rho_k * P_k^T P_k * x
    % Each element appears 4 times in the 4 unfoldings
    y_out = rho_k * 4 * x;
    
    % Term 2: rho_m/m * A^T A * x
    % A^T A x = sum_i <A_i, T> * A_i
    for i = 1:length(A_cells)
        Ai_tensor = compute_Ai_tensor(A_cells{i});
        inner_prod = tensor_inner_product(Ai_tensor, T);
        y_out = y_out + (rho_m / m) * inner_prod * Ai_tensor(:);
    end
end

function b = compute_rhs_v2(W, Z, A_cells, y, u, rho_k, rho_m, tensor_dims, m)
    % Compute right-hand side for T update
    % rhs = rho_m * A^T (y + u) / sqrt(m) + sum_k rho_k * P_k^T vec(W_k + Z_k)
    
    b = zeros(prod(tensor_dims), 1);
    
    % Term 1: sum_k rho_k * P_k^T vec(W_k + Z_k)
    for k = 1:4
        W_plus_Z = W{k} + Z{k};
        T_folded = tensor_mode_fold(W_plus_Z, k, tensor_dims);
        b = b + rho_k * T_folded(:);
    end
    
    % Term 2: rho_m * A^T (y + u) / sqrt(m)
    y_plus_u = y + u;
    for i = 1:length(y_plus_u)
        Ai_tensor = compute_Ai_tensor(A_cells{i});
        b = b + (rho_m / sqrt(m)) * y_plus_u(i) * Ai_tensor(:);
    end
end

function W_new = singular_value_thresholding(M, tau)
    % Singular Value Thresholding: SVT_tau(M)
    % W = U * diag(max(sigma - tau, 0)) * V^T
    [U, S, V] = svd(M, 'econ');
    s = diag(S);
    s_new = max(s - tau, 0);  % Soft thresholding
    W_new = U * diag(s_new) * V';
end

function T_mat = tensor_mode_unfold(T, mode)
    % Mode-k unfolding (matricization) of tensor T
    sz = size(T);
    n_modes = length(sz);
    
    % Permute so mode is first
    perm = [mode, 1:mode-1, mode+1:n_modes];
    T_perm = permute(T, perm);
    
    % Reshape to matrix
    T_mat = reshape(T_perm, sz(mode), []);
end

function T = tensor_mode_fold(T_mat, mode, tensor_dims)
    % Inverse of tensor_mode_unfold: fold matrix back to tensor
    n_modes = length(tensor_dims);
    
    % Compute permuted dimensions
    perm = [mode, 1:mode-1, mode+1:n_modes];
    dims_perm = tensor_dims(perm);
    
    % Reshape matrix to permuted tensor
    T_perm = reshape(T_mat, dims_perm);
    
    % Inverse permutation
    inv_perm(perm) = 1:n_modes;
    T = permute(T_perm, inv_perm);
end

function Ai_tensor = compute_Ai_tensor(Ai)
    % Compute 4th-order tensor A_i ⊗ A_i
    d = size(Ai, 1);
    Ai_vec = Ai(:);
    Ai_outer = Ai_vec * Ai_vec';  % (d²×d²) outer product
    Ai_tensor = reshape(Ai_outer, [d, d, d, d]);
end

function inner_prod = tensor_inner_product(A, B)
    % Compute <A, B> = sum_{i,j,k,l} A(i,j,k,l) * B(i,j,k,l)
    inner_prod = sum(A(:) .* B(:));
end

function sizes = compute_unfolding_sizes(d)
    % Compute sizes of mode-k unfoldings for d×d×d×d tensor
    sizes = cell(1, 4);
    sizes{1} = [d, d^3];
    sizes{2} = [d, d^3];
    sizes{3} = [d, d^3];
    sizes{4} = [d, d^3];
end
