function [X0, U0, history] = initialize_tensor_nuclear_norm_v2(y, operator, d1, d2, params)
% INITIALIZE_TENSOR_NUCLEAR_NORM_V2 Initialization using tensor nuclear norm minimization (v2)
%
% This function provides an initial estimate for matrix recovery by running
% a few iterations of tensor nuclear norm minimization (v2 solver with separate
% penalty parameters), then extracting the matrix. This can be used as initialization
% for non-convex local refinement algorithms.
%
% Inputs:
%   y        - Measurement vector (m × 1)
%   operator - Struct with measurement operators:
%              .A: Forward operator @(X) A*X(:) for matrix
%              .A_star: Adjoint operator @(y) reshape(A'*y, [d1,d2])
%              .A_cells: Cell array {A_1, ..., A_m} of d×d matrices (optional)
%   d1       - Matrix row dimension
%   d2       - Matrix column dimension
%   params   - Struct with optional fields:
%              .r (or .rank): Target rank for extraction (default: estimated)
%              .T_power (or .max_iter): Number of iterations to run (default: 10)
%              .verbose: Verbosity level (default: 0)
%              .normalize: Normalize output (default: true)
%              .rho_k: Penalty parameter for unfolding constraints (default: 0.1)
%              .rho_m: Penalty parameter for measurement constraints (default: 1.0)
%              .lambda: Weight vector for mode nuclear norms (default: [1,1,1,1])
%
% Note: This initialization uses the v2 solver with separate penalty parameters:
%       - rho_k: penalty for unfolding constraints W_k = T_(k)
%       - rho_m: penalty for measurement constraints
%       - lambda: weights for each mode nuclear norm
%       - pcg_tol = 1e-3, pcg_maxit = 100 (from v2 defaults)
%
% Outputs:
%   X0      - Initial estimate of the d×d matrix
%   U0      - Empty (for compatibility with other initialization functions)
%   history - Struct with convergence information:
%             .method: 'tensor_nuclear_norm_v2'
%             .iterations: actual iterations run
%             .obj_values: objective function history
%             .primal_residuals: ADMM primal residual history
%             .measurement_residuals: measurement residual history
%             .rank: rank of recovered matrix
%             .converged: convergence flag
%
% Example:
%   params = struct('r', 1, 'T_power', 10, 'verbose', 1);
%   [X0, U0, history] = initialize_tensor_nuclear_norm_v2(y, operator, 20, 20, params);

% Validate symmetric case
if d1 ~= d2
    error('Tensor nuclear norm v2 initialization requires symmetric matrices: d1 must equal d2');
end
d = d1;

% Extract parameters from params struct
if nargin < 5 || isempty(params)
    params = struct();
end

rank_param = get_param(params, 'r', []);
if isempty(rank_param)
    rank_param = get_param(params, 'rank', []);  % Alternative field name
end
max_iter = get_param(params, 'T_power', 10);
verbose = get_param(params, 'verbose', 0);
normalize = get_param(params, 'normalize', true);

% V2-specific parameters
rho_k = get_param(params, 'rho_k', 0.1);
rho_m = get_param(params, 'rho_m', 1.0);
lambda_vec = get_param(params, 'lambda', [1, 1, 1, 1]);

% Estimate rank if not provided
if isempty(rank_param)
    % Heuristic: assume low rank, default to small value
    rank_param = min(5, floor(d/4));
    if verbose >= 1
        fprintf('[Init-TensorNuclear-v2] Estimated rank: %d\n', rank_param);
    end
end

if verbose >= 1
    fprintf('[Init-TensorNuclear-v2] Starting initialization with %d iterations\n', max_iter);
    fprintf('[Init-TensorNuclear-v2] Using v2 solver parameters:\n');
    fprintf('  rho_k = %.2e (unfolding penalty)\n', rho_k);
    fprintf('  rho_m = %.2e (measurement penalty)\n', rho_m);
    fprintf('  lambda = [%.2f, %.2f, %.2f, %.2f] (mode weights)\n', lambda_vec);
    fprintf('[Init-TensorNuclear-v2] Target rank: %d\n', rank_param);
end

% Convert phase retrieval measurements to tensor measurements
% Phase retrieval: y_i = |<A_i, X>| / sqrt(m)
% Tensor formulation: y_tensor_i = <A_i ⊗ A_i, X ⊗ X> = (<A_i, X>)^2 / m
% So: y_tensor = y.^2 (since y already includes the 1/sqrt(m) factor)
m = length(y);
y_tensor = y.^2;  % Convert amplitude to squared measurements

if verbose >= 1
    fprintf('[Init-TensorNuclear-v2] Converted %d phase retrieval measurements to tensor format\n', m);
    fprintf('[Init-TensorNuclear-v2] Measurement range: [%.2e, %.2e]\n', min(y_tensor), max(y_tensor));
end

% Extract A_cells from operator if not already present
if ~isfield(operator, 'A_cells')
    if verbose >= 1
        fprintf('[Init-TensorNuclear-v2] Extracting measurement matrices from operator...\n');
    end
    m = length(y);
    n = d * d;
    A_matrix = zeros(m, n);
    for j = 1:n
        e_j = zeros(n, 1);
        e_j(j) = 1;
        E_j = reshape(e_j, [d, d]);
        A_matrix(:, j) = operator.A(E_j);
    end
    
    % Create cell array of measurement matrices
    A_cells = cell(m, 1);
    for i = 1:m
        Ai = reshape(A_matrix(i, :), [d, d]);
        A_cells{i} = (Ai + Ai') / 2;  % Symmetrize
    end
    operator.A_cells = A_cells;
end

% Run tensor nuclear norm minimization with v2 solver
try
    [X0, info] = solve_tensor_nuclear_norm_v2(operator, y_tensor, d, ...
        'max_iter', max_iter, ...
        'rank', rank_param, ...
        'rho_k', rho_k, ...
        'rho_m', rho_m, ...
        'lambda', lambda_vec, ...
        'verbose', max(0, verbose-1));  % Reduce verbosity by 1 level
    
    % Check for NaN or Inf
    if any(isnan(X0(:))) || any(isinf(X0(:)))
        error('Tensor nuclear norm v2 solver returned NaN or Inf values');
    end
    
    if verbose >= 1
        fprintf('[Init-TensorNuclear-v2] Initialization complete after %d iterations\n', info.iter);
        fprintf('[Init-TensorNuclear-v2] Final objective: %.6e\n', info.obj_values(end));
        fprintf('[Init-TensorNuclear-v2] Final primal residual: %.6e\n', ...
                info.primal_residuals(end));
        fprintf('[Init-TensorNuclear-v2] Final measurement residual: %.6e\n', ...
                info.measurement_residuals(end));
        fprintf('[Init-TensorNuclear-v2] Recovered rank: %d\n', rank(X0, 1e-6));
    end
    
catch ME
    % Fallback to random initialization if tensor nuclear norm fails
    warning('InitTensorNuclearV2:SolverFailed', 'Tensor nuclear norm v2 initialization failed: %s', ME.message);
    if verbose >= 1
        fprintf('[Init-TensorNuclear-v2] Falling back to random initialization\n');
    end
    
    % Random symmetric matrix
    X0 = randn(d, d);
    X0 = (X0 + X0') / 2;
    X0 = X0 / norm(X0, 'fro');
    
    % Create minimal info struct
    info = struct();
    info.iter = 0;
    info.obj_values = [];
    info.primal_residuals = [];
    info.measurement_residuals = [];
    info.times = [];
    info.total_time = 0;
    info.converged = false;
end

% Normalize if requested
if normalize
    X0_norm = norm(X0, 'fro');
    if X0_norm > 1e-10  % Avoid division by zero
        X0 = X0 / X0_norm;
    else
        warning('X0 norm is too small, using random initialization');
        X0 = randn(d, d);
        X0 = (X0 + X0') / 2;
        X0 = X0 / norm(X0, 'fro');
    end
    if verbose >= 1
        fprintf('[Init-TensorNuclear-v2] Normalized output to unit Frobenius norm\n');
    end
end

% Ensure symmetry
X0 = (X0 + X0') / 2;

% Final validation
if any(isnan(X0(:))) || any(isinf(X0(:)))
    warning('X0 still contains NaN/Inf after processing, using identity matrix');
    X0 = eye(d) / sqrt(d);
end

% Apply projected power method iterations for refinement (optional)
apply_power_refinement = get_param(params, 'apply_power_refinement', false);
if apply_power_refinement
    num_power_iters = get_param(params, 'num_power_refinements', 10);
    if verbose >= 1
        fprintf('[Init-TensorNuclear-v2] Applying %d projected power method iterations for refinement...\n', ...
                num_power_iters);
    end
    
    % Prepare power method parameters
    power_params = params;
    power_params.T_power = num_power_iters;
    power_params.prefunc = @(y) y;  % No preprocessing (y already in correct format)
    power_params.Init = X0(:);  % Initialize with TNN result
    
    % Apply projection if rank is specified
    if ~isempty(rank_param)
        power_params.projection = @(X) project_rank_r_helper(X, rank_param);
    else
        power_params.projection = [];
    end
    
    % Run power method
    [X0_refined, ~, ~] = initialize_power_method(y, operator, d, d, power_params);
    
    % Use refined result
    X0 = X0_refined;
    
    % Ensure symmetry after power method
    X0 = (X0 + X0') / 2;
    
    % Normalize
    if normalize
        X0 = X0 / norm(X0, 'fro');
    end
    
    if verbose >= 1
        fprintf('[Init-TensorNuclear-v2] Power method refinement complete\n');
    end
end

if verbose >= 1
    fprintf('[Init-TensorNuclear-v2] Output: %dx%d matrix, rank=%d, norm=%.4f\n', ...
            size(X0,1), size(X0,2), rank(X0, 1e-6), norm(X0, 'fro'));
end

% Set U0 (empty for compatibility with other initialization functions)
U0 = [];

% Populate history struct
history = struct();
history.method = 'tensor_nuclear_norm_v2';
history.iterations = info.iter;
history.max_iter = max_iter;
history.obj_values = info.obj_values;
history.primal_residuals = info.primal_residuals;
history.measurement_residuals = info.measurement_residuals;
history.times = info.times;
history.total_time = info.total_time;
history.rank = rank(X0, 1e-6);
history.converged = info.converged;
history.rho_k = rho_k;
history.rho_m = rho_m;
history.lambda = lambda_vec;

% Include errors if available
if isfield(info, 'errors')
    history.errors = info.errors;
end

end

%% Helper Functions
function value = get_param(params, field, default)
    % Get parameter from struct with default value
    if isfield(params, field)
        value = params.(field);
    else
        value = default;
    end
end

function X_proj = project_rank_r_helper(X, r)
    % Project matrix X onto the set of symmetric rank-r matrices
    % Input:
    %   X - Input matrix (d x d)
    %   r - Target rank
    % Output:
    %   X_proj - Projected symmetric rank-r matrix
    
    % Handle vector input (from power method)
    if isvector(X)
        d = sqrt(length(X));
        X = reshape(X, [d, d]);
    end
    
    % Symmetrize
    X_sym = (X + X') / 2;
    
    % SVD and truncate to rank r
    [U, S, ~] = svd(X_sym);
    U_r = U(:, 1:r);
    S_r = S(1:r, 1:r);
    
    % Reconstruct symmetric rank-r matrix
    X_proj = U_r * S_r * U_r';
end
