function [X0, U0, history] = initialize_tensor_nuclear_norm(y, operator, d1, d2, params)
% INITIALIZE_TENSOR_NUCLEAR_NORM Initialization using tensor nuclear norm minimization
%
% This function provides an initial estimate for matrix recovery by running
% a few iterations of tensor nuclear norm minimization (convex relaxation),
% then extracting the matrix. This can be used as initialization for
% non-convex local refinement algorithms.
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
%
% Note: This initialization uses the default solver parameters:
%       - over_relax = 1.5 (ADMM over-relaxation)
%       - pcg_tol = 1e-2, pcg_maxit = 30 (rough PCG solve)
%       - lambda = 1.0, rho = 1.0 (ADMM parameters)
%
% Outputs:
%   X0      - Initial estimate of the d×d matrix
%   U0      - Empty (for compatibility with other initialization functions)
%   history - Struct with convergence information:
%             .method: 'tensor_nuclear_norm'
%             .iterations: actual iterations run
%             .obj_values: objective function history
%             .constraint_violations: constraint violation history
%             .primal_residuals: ADMM primal residual history
%             .dual_residuals: ADMM dual residual history
%             .rank: rank of recovered matrix
%             .converged: convergence flag
%
% Example:
%   [X0, U0, history] = initialize_tensor_nuclear_norm(y, operator, 20, 20, params);

% Validate symmetric case
if d1 ~= d2
    error('Tensor nuclear norm initialization requires symmetric matrices: d1 must equal d2');
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

% Estimate rank if not provided
if isempty(rank_param)
    % Heuristic: assume low rank, default to small value
    rank_param = min(5, floor(d/4));
    if verbose >= 1
        fprintf('[Init-TensorNuclear] Estimated rank: %d\n', rank_param);
    end
end

if verbose >= 1
    fprintf('[Init-TensorNuclear] Starting initialization with %d iterations\n', max_iter);
    fprintf('[Init-TensorNuclear] Using default solver parameters (over-relax α=1.5, rough PCG)\n');
    fprintf('[Init-TensorNuclear] Target rank: %d\n', rank_param);
end

% Convert phase retrieval measurements to tensor measurements
% Phase retrieval: y_i = |<A_i, X>| / sqrt(m)
% Tensor formulation: y_tensor_i = <A_i ⊗ A_i, X ⊗ X> = (<A_i, X>)^2 / m
% So: y_tensor = y.^2 (since y already includes the 1/sqrt(m) factor)
m = length(y);
y_tensor = y.^2;  % Convert amplitude to squared measurements

if verbose >= 1
    fprintf('[Init-TensorNuclear] Converted %d phase retrieval measurements to tensor format\n', m);
    fprintf('[Init-TensorNuclear] Measurement range: [%.2e, %.2e]\n', min(y_tensor), max(y_tensor));
end

% Extract A_cells from operator if not already present
if ~isfield(operator, 'A_cells')
    if verbose >= 1
        fprintf('[Init-TensorNuclear] Extracting measurement matrices from operator...\n');
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

% Run tensor nuclear norm minimization with default solver parameters
% The solver now uses: over_relax=1.5, pcg_tol=1e-2, pcg_maxit=30
% These defaults provide good balance between speed and accuracy for initialization
try
    [X0, info] = solve_tensor_nuclear_norm(operator, y_tensor, d, ...
        'max_iter', max_iter, ...
        'rank', rank_param, ...
        'verbose', max(0, verbose-1));  % Reduce verbosity by 1 level
    
    % Check for NaN or Inf
    if any(isnan(X0(:))) || any(isinf(X0(:)))
        error('Tensor nuclear norm solver returned NaN or Inf values');
    end
    
    if verbose >= 1
        fprintf('[Init-TensorNuclear] Initialization complete after %d iterations\n', info.iter);
        fprintf('[Init-TensorNuclear] Final objective: %.6e\n', info.obj_values(end));
        fprintf('[Init-TensorNuclear] Final constraint violation: %.6e\n', ...
                info.constraint_violations(end));
        fprintf('[Init-TensorNuclear] Recovered rank: %d\n', rank(X0, 1e-6));
    end
    
catch ME
    % Fallback to random initialization if tensor nuclear norm fails
    warning('InitTensorNuclear:SolverFailed', 'Tensor nuclear norm initialization failed: %s', ME.message);
    if verbose >= 1
        fprintf('[Init-TensorNuclear] Falling back to random initialization\n');
    end
    
    % Random symmetric matrix
    X0 = randn(d, d);
    X0 = (X0 + X0') / 2;
    X0 = X0 / norm(X0, 'fro');
    
    % Create minimal info struct
    info = struct();
    info.iter = 0;
    info.obj_values = [];
    info.constraint_violations = [];
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
        fprintf('[Init-TensorNuclear] Normalized output to unit Frobenius norm\n');
    end
end

% Ensure symmetry
X0 = (X0 + X0') / 2;

% Final validation
if any(isnan(X0(:))) || any(isinf(X0(:)))
    warning('X0 still contains NaN/Inf after processing, using identity matrix');
    X0 = eye(d) / sqrt(d);
end

%% Apply 10 extra projected power method iterations for refinement
if verbose >= 1
    fprintf('[Init-TensorNuclear] Applying 10 projected power method iterations for refinement...\n');
end

% Prepare power method parameters
power_params = params;
power_params.T_power = 0;  % Number of power iterations
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
    fprintf('[Init-TensorNuclear] Power method refinement complete\n');
    fprintf('[Init-TensorNuclear] Output: %dx%d matrix, rank=%d, norm=%.4f\n', ...
            size(X0,1), size(X0,2), rank(X0, 1e-6), norm(X0, 'fro'));
end

% Set U0 (empty for compatibility with other initialization functions)
U0 = [];

% Populate history struct
history = struct();
history.method = 'tensor_nuclear_norm';
history.iterations = info.iter;
history.max_iter = max_iter;
history.obj_values = info.obj_values;
history.constraint_violations = info.constraint_violations;
history.times = info.times;
history.total_time = info.total_time;
history.rank = rank(X0, 1e-6);
history.converged = info.converged;

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
