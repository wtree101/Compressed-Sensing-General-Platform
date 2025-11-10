function [X_recovered, info] = solve_tensor_nuclear_norm(operator, y, d, varargin)
% SOLVE_TENSOR_NUCLEAR_NORM Tensor nuclear norm minimization for phase retrieval
%
% This solver recovers a symmetric matrix X from measurements y using
% tensor nuclear norm minimization on the lifted tensor T = X ⊗ X.
%
% Problem formulation:
%   min_T  sum_{k=1}^4 ||T_(k)||_*
%   s.t.   y_i = <A_i, T> / sqrt(m), for all i
%
% where:
%   A_i = A_i ⊗ A_i (4th-order measurement tensors)
%   T = X ⊗ X (4th-order target tensor)
%   T_(k) is the mode-k unfolding (matricization) of tensor T
%   ||·||_* is the nuclear norm (sum of singular values)
%   Measurements are scaled by 1/sqrt(m)
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
%   'lambda'      - Regularization parameter (default: 1.0)
%   'rho'         - ADMM penalty parameter (default: 1.0)
%   'verbose'     - Verbosity level: 0=silent, 1=basic, 2=detailed (default: 1)
%   'X_true'      - Ground truth for computing error (optional)
%   'use_admm'    - Use ADMM algorithm (default: true)
%   'nuclear_weight' - Weight vector for mode nuclear norms [w1,w2,w3,w4] (default: [1,1,1,1])
%   'rank'        - Target rank for matrix extraction (default: rank of X_true or 5)
%
% Outputs:
%   X_recovered - Recovered d×d matrix
%   info        - Struct with solver information:
%                 .obj_values: Objective values per iteration
%                 .errors: Reconstruction errors (if X_true provided)
%                 .times: Computation time per iteration
%                 .iter: Total iterations
%                 .converged: Convergence flag
%
% Algorithm: ADMM (Alternating Direction Method of Multipliers)
%   We use ADMM to solve:
%   min_{T,Z_k}  sum_k w_k * ||Z_k||_*
%   s.t.  y_i = <A_i, T> / sqrt(m), Z_k = T_(k), for all i,k
%
% Example:
%   [X_rec, info] = solve_tensor_nuclear_norm(operator, y, 20);
%   [X_rec, info] = solve_tensor_nuclear_norm(operator, y, 20, 'lambda', 0.1, 'max_iter', 500);

% Parse inputs
p = inputParser;
addRequired(p, 'operator');
addRequired(p, 'y');
addRequired(p, 'd');
addParameter(p, 'max_iter', 1000, @isnumeric);
addParameter(p, 'tol', 1e-6, @isnumeric);
addParameter(p, 'lambda', 1.0, @isnumeric);
addParameter(p, 'rho', 0.1, @isnumeric);
addParameter(p, 'verbose', 1, @isnumeric);
addParameter(p, 'X_true', [], @isnumeric);
addParameter(p, 'use_admm', true, @islogical);
addParameter(p, 'nuclear_weight', [1, 1, 1, 1], @isnumeric);
addParameter(p, 'rank', [], @isnumeric);  % Target rank for extraction

% Robustness parameters
addParameter(p, 'use_spectral_init', false, @islogical); % Warm start via Kronecker adjoint (disabled by default)
addParameter(p, 'over_relax', 1.0, @isnumeric);          % ADMM over-relaxation α ∈ [1, 2] (disabled by default)
addParameter(p, 'adapt_rho', false, @islogical);         % Adaptive rho (residual balancing) (disabled by default)
addParameter(p, 'rho_tau', 2.0, @isnumeric);             % Rho scaling factor
addParameter(p, 'rho_mu', 10.0, @isnumeric);             % Residual ratio threshold
addParameter(p, 'restart_if_stalled', false, @islogical); % Restart mechanism (disabled by default)
addParameter(p, 'stall_check_iter', 15, @isnumeric);     % Check for stall at this iteration
addParameter(p, 'stall_norm_tol', 1e-8, @isnumeric);     % Tensor norm threshold for stall
addParameter(p, 'pcg_tol', 1e-3, @isnumeric);            % PCG tolerance (rough solve)
addParameter(p, 'pcg_maxit', 100, @isnumeric);             % PCG max iterations (minimal solve)
addParameter(p, 'use_pcg_precond', false, @islogical);   % Use Jacobi preconditioner for PCG (disabled by default)

parse(p, operator, y, d, varargin{:});

max_iter = p.Results.max_iter;
tol = p.Results.tol;
lambda = p.Results.lambda;
rho = p.Results.rho;
verbose = p.Results.verbose;
X_true = p.Results.X_true;
use_admm = p.Results.use_admm;
w_nuclear = p.Results.nuclear_weight(:)';
target_rank = p.Results.rank;

% Robustness parameters
use_spectral_init = p.Results.use_spectral_init;
alpha_over = p.Results.over_relax;
adapt_rho = p.Results.adapt_rho;
rho_tau = p.Results.rho_tau;
rho_mu = p.Results.rho_mu;
restart_if_stalled = p.Results.restart_if_stalled;
stall_check_iter = p.Results.stall_check_iter;
stall_norm_tol = p.Results.stall_norm_tol;
pcg_tol = p.Results.pcg_tol;
pcg_maxit = p.Results.pcg_maxit;
use_pcg_precond = p.Results.use_pcg_precond;

% Validate inputs
if ~isfield(operator, 'A_cells')
    error('operator must have A_cells field');
end
m = length(y);
if length(operator.A_cells) ~= m
    error('Number of measurement matrices must equal length of y');
end

% Print header
if verbose >= 1
    fprintf('\n=== Tensor Nuclear Norm Minimization ===\n');
    fprintf('Problem size: d=%d, measurements m=%d\n', d, m);
    fprintf('Tensor size: %d×%d×%d×%d\n', d, d, d, d);
    fprintf('Algorithm: %s\n', ternary(use_admm, 'ADMM', 'Subgradient'));
    fprintf('Parameters: lambda=%.2e, rho=%.2e\n', lambda, rho);
    fprintf('Nuclear norm weights: [%.2f, %.2f, %.2f, %.2f]\n', w_nuclear);
    fprintf('Max iterations: %d, tolerance: %.2e\n', max_iter, tol);
    fprintf('Over-relax α: %.2f, PCG: tol=%.0e, maxit=%d, precond=%s\n', ...
            alpha_over, pcg_tol, pcg_maxit, mat2str(use_pcg_precond));
    if use_spectral_init || adapt_rho || restart_if_stalled
        fprintf('Optional: Spectral init: %s, Adaptive rho: %s, Restart: %s\n', ...
                mat2str(use_spectral_init), mat2str(adapt_rho), mat2str(restart_if_stalled));
    end
    fprintf('\n');
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
info.constraint_violations = zeros(max_iter, 1);
info.times = zeros(max_iter, 1);
info.converged = false;  % Initialize convergence flag
if ~isempty(X_true)
    info.errors = zeros(max_iter, 1);
end

% ADMM variables
Z = cell(1, 4);  % Auxiliary variables for each mode unfolding
U = cell(1, 4);  % Dual variables (scaled Lagrange multipliers)

% Initialize Z_k and U_k for each mode
unfolding_sizes = compute_unfolding_sizes(d);
for k = 1:4
    % Initialize Z close to T for warm start
    Z{k} = tensor_mode_unfold(T, k);
    U{k} = zeros(unfolding_sizes{k});
end

% Track rho history
info.rho_hist = zeros(max_iter, 1);

start_time = tic;

%% Main ADMM iteration
for iter = 1:max_iter
    iter_start = tic;
    
    % Store previous Z for over-relaxation and dual residual
    Z_prev = Z;
    
    % Step 1: Update T (solve measurement constraint with regularization)
    % min_T  (rho/2) * sum_k ||T_(k) - Z_k + U_k||_F^2
    %        + (lambda/2) * sum_i (y_i - <A_i, T> / sqrt(m))^2
    T = update_T_least_squares(T, Z, U, operator.A_cells, y, rho, lambda, tensor_dims, m, pcg_tol, pcg_maxit, use_pcg_precond);
    
    % Step 2: Over-relaxed Z-update (proximal operator of nuclear norm)
    for k = 1:4
        T_k = tensor_mode_unfold(T, k);
        % Over-relaxation: T̂_k = α T_k + (1-α) Z_prev{k}
        T_hat_k = alpha_over * T_k + (1 - alpha_over) * Z_prev{k};
        Z{k} = prox_nuclear_norm(T_hat_k + U{k}, w_nuclear(k) / rho);
    end
    
    % Step 3: Update dual variables U_k (using over-relaxed variable)
    for k = 1:4
        T_k = tensor_mode_unfold(T, k);
        T_hat_k = alpha_over * T_k + (1 - alpha_over) * Z_prev{k};
        U{k} = U{k} + (T_hat_k - Z{k});
    end
    
    % Compute objective and constraint violation
    obj_val = 0;
    for k = 1:4
        obj_val = obj_val + w_nuclear(k) * sum(svd(tensor_mode_unfold(T, k)));
    end
    
    % Measurement constraint violation
    constraint_viol = 0;
    for i = 1:m
        Ai_tensor = compute_Ai_tensor(operator.A_cells{i});
        constraint_viol = constraint_viol + (y(i) - tensor_inner_product(Ai_tensor, T) / sqrt(m))^2;
    end
    constraint_viol = sqrt(constraint_viol / m);
    
    % Store iteration info
    info.obj_values(iter) = obj_val;
    info.constraint_violations(iter) = constraint_viol;
    info.times(iter) = toc(iter_start);
    info.rho_hist(iter) = rho;
    
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
        obj_change = abs(info.obj_values(iter) - info.obj_values(iter-1)) / (abs(info.obj_values(iter-1)) + 1e-10);
        
        % Primal and dual residuals
        primal_residual = 0;
        dual_residual = 0;
        for k = 1:4
            T_k = tensor_mode_unfold(T, k);
            T_hat_k = alpha_over * T_k + (1 - alpha_over) * Z_prev{k};
            primal_residual = primal_residual + norm(T_hat_k - Z{k}, 'fro')^2;
            dual_residual = dual_residual + rho^2 * norm(Z{k} - Z_prev{k}, 'fro')^2;
        end
        primal_residual = sqrt(primal_residual);
        dual_residual = sqrt(dual_residual);
        
        % Adaptive rho (residual balancing)
        if adapt_rho
            if primal_residual > rho_mu * dual_residual
                rho_new = rho * rho_tau;
            elseif dual_residual > rho_mu * primal_residual
                rho_new = rho / rho_tau;
            else
                rho_new = rho;
            end
            if rho_new ~= rho
                % Scale U to maintain consistency
                scale = rho / rho_new;
                for k = 1:4
                    U{k} = scale * U{k};
                end
                rho = rho_new;
            end
        end
        
        converged = (obj_change < tol) && (primal_residual < tol) && (constraint_viol < tol);
        
        if verbose >= 2 || (verbose >= 1 && mod(iter, 10) == 0)
            fprintf('Iter %4d: obj=%.6e, |r_p|=%.2e, |r_d|=%.2e, |constr|=%.2e, rho=%.2f', ...
                    iter, obj_val, primal_residual, dual_residual, constraint_viol, rho);
            if ~isempty(X_true)
                fprintf(', err=%.6e', info.errors(iter));
            end
            fprintf('\n');
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
            fprintf('Iter %4d: obj=%.6e, |constr|=%.2e, rho=%.2f', iter, obj_val, constraint_viol, rho);
            if ~isempty(X_true)
                fprintf(', err=%.6e', info.errors(iter));
            end
            fprintf('\n');
        end
    end
    
    % Restart mechanism if stalled at small scale
    if restart_if_stalled && iter == stall_check_iter
        T_norm = norm(T(:));
        if T_norm < stall_norm_tol
            if verbose >= 1
                fprintf('[Restart] ||T|| = %.2e is too small. Reinitializing from adjoint...\n', T_norm);
            end
            % Re-initialize from Kronecker adjoint
            T = zeros(tensor_dims);
            inv_sqrt_m = 1 / sqrt(m);
            for i = 1:m
                if y(i) ~= 0
                    Ai_tensor = compute_Ai_tensor(operator.A_cells{i});
                    T = T + (inv_sqrt_m * y(i)) * Ai_tensor;
                end
            end
            % Reset Z to match T
            for k = 1:4
                Z{k} = tensor_mode_unfold(T, k);
            end
            % Increase lambda to enforce measurements more
            lambda = max(lambda, 10);
            if verbose >= 1
                fprintf('[Restart] New ||T|| = %.2e, lambda increased to %.2e\n', norm(T(:)), lambda);
            end
        end
    end
end

total_time = toc(start_time);

% Trim unused entries
info.obj_values = info.obj_values(1:iter);
info.constraint_violations = info.constraint_violations(1:iter);
info.times = info.times(1:iter);
info.rho_hist = info.rho_hist(1:iter);
if ~isempty(X_true)
    info.errors = info.errors(1:iter);
end
info.iter = iter;
info.total_time = total_time;
info.final_rho = rho;
% info.converged is already set during iteration

% Extract matrix from tensor
% Determine rank for extraction
if isempty(target_rank)
    if ~isempty(X_true)
        r_extract = rank(X_true, 1e-6);
    else
        r_extract = min(5, d);  % Default rank
    end
else
    r_extract = target_rank;
end
params_extract = struct('r', r_extract, 'method', 'eig', 'verbose', (verbose >= 2));
X_recovered = extract_matrix_from_tensor(T, params_extract);

% Final summary
if verbose >= 1
    fprintf('\n=== Solver Summary ===\n');
    fprintf('Total iterations: %d\n', iter);
    fprintf('Total time: %.4f seconds\n', total_time);
    fprintf('Final objective: %.6e\n', info.obj_values(end));
    fprintf('Final constraint violation: %.6e\n', info.constraint_violations(end));
    fprintf('Converged: %s\n', mat2str(info.converged));
    if ~isempty(X_true)
        fprintf('Final reconstruction error: %.6e\n', info.errors(end));
    end
    fprintf('Recovered matrix rank: %d\n', rank(X_recovered, 1e-6));
    fprintf('======================\n\n');
end

end

%% Helper Functions

function T_new = update_T_least_squares(T, Z, U, A_cells, y, rho, lambda, tensor_dims, m, pcg_tol, pcg_maxit, use_precond)
    % Update T by solving least squares problem
    % min_T  (rho/2) * sum_k ||T_(k) - Z_k + U_k||_F^2
    %        + (lambda/2) * sum_i (y_i - <A_i, T> / sqrt(m))^2
    %
    % This is solved using conjugate gradient with optional scalar preconditioner
    
    % Use conjugate gradient for large problems
    T_vec = T(:);
    
    % Define linear operator: A(T) for ADMM objective
    A_op = @(x) apply_T_operator(x, A_cells, Z, U, rho, lambda, tensor_dims, m);
    
    % Define right-hand side
    b = compute_rhs(Z, U, A_cells, y, rho, lambda, tensor_dims, m);
    
    % Optionally use cheap scalar Jacobi preconditioner: M ≈ sigma*I
    % sigma ≈ rho*4 + (lambda/m) * mean_i(||Ai||_F^2)
    % This stabilizes PCG with minimal overhead
    if use_precond
        if ~isempty(A_cells) && length(A_cells) > 0
            % Sample first few for speed (or cache this)
            n_sample = min(10, length(A_cells));
            Ai_frob2_sample = 0;
            for i = 1:n_sample
                Ai_frob2_sample = Ai_frob2_sample + norm(A_cells{i}, 'fro')^2;
            end
            mean_Ai_frob2 = Ai_frob2_sample / n_sample;
            sigma = rho * 4 + (lambda / m) * mean_Ai_frob2;
        else
            sigma = rho * 4;  % fallback
        end
        M = @(z) (1 / sigma) * z;
        [T_vec_new, ~] = pcg(A_op, b, pcg_tol, pcg_maxit, M, [], T_vec);
    else
        % No preconditioner
        [T_vec_new, ~] = pcg(A_op, b, pcg_tol, pcg_maxit, [], [], T_vec);
    end
    
    T_new = reshape(T_vec_new, tensor_dims);
end

function y_out = apply_T_operator(x, A_cells, ~, ~, rho, lambda, tensor_dims, m)
    % Apply the linear operator for T update
    T = reshape(x, tensor_dims);
    
    % ADMM term: rho * sum_k T_(k)
    y_out = rho * 4 * x;  % Simplified: each element appears 4 times in unfoldings
    
    % Measurement term: lambda * sum_i (<A_i, T> / sqrt(m)) * A_i / sqrt(m)
    % This accounts for the scaling: y_i = <A_i, T> / sqrt(m)
    for i = 1:length(A_cells)
        Ai_tensor = compute_Ai_tensor(A_cells{i});
        inner_prod = tensor_inner_product(Ai_tensor, T);
        y_out = y_out + (lambda / m) * inner_prod * Ai_tensor(:);
    end
end

function b = compute_rhs(Z, U, A_cells, y, rho, lambda, tensor_dims, m)
    % Compute right-hand side for T update
    b = zeros(prod(tensor_dims), 1);
    
    % ADMM term: rho * sum_k (Z_k - U_k) folded back
    for k = 1:4
        T_k_target = Z{k} - U{k};
        T_folded = tensor_mode_fold(T_k_target, k, tensor_dims);
        b = b + rho * T_folded(:);
    end
    
    % Measurement term: lambda * sum_i y_i * A_i / sqrt(m)
    % This accounts for the scaling: y_i = <A_i, T> / sqrt(m)
    for i = 1:length(y)
        Ai_tensor = compute_Ai_tensor(A_cells{i});
        b = b + (lambda / sqrt(m)) * y(i) * Ai_tensor(:);
    end
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

function Z_new = prox_nuclear_norm(A, tau)
    % Proximal operator of nuclear norm: prox_{tau*||·||_*}(A)
    % Solution: singular value soft thresholding
    [U, S, V] = svd(A, 'econ');
    s = diag(S);
    s_new = max(s - tau, 0);  % Soft thresholding
    Z_new = U * diag(s_new) * V';
end

function Ai_tensor = compute_Ai_tensor(Ai)
    % Compute 4th-order tensor A_i ⊗ A_i
    % For efficiency, we use the Kronecker product structure
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

function result = ternary(condition, true_val, false_val)
    % Ternary operator
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
