function [X0, U0, history] = initialize_tensor_lift(y, operator, d1, d2, params)
% INITIALIZE_TENSOR_LIFT Tensor-lifted initialization for matrix recovery
%
% This function performs initialization by lifting the matrix recovery problem
% to a tensor space, running tensor PGD with projection, and extracting the matrix.
% Uses the formulation X = UU^T viewed as fourth-order tensor T = X ⊗ X.
%
% Inputs:
%   y        - Measurement vector (m x 1)
%   operator - Struct with fields:
%              .A: Forward operator @(X) A*X(:) for matrix
%              .A_star: Adjoint operator @(y) reshape(A'*y, [d1,d2])
%   d1       - Matrix row dimension
%   d2       - Matrix column dimension
%   params   - Struct with optional fields:
%              .T_power: Number of tensor PGD iterations (default: 5)
%              .mu: Step size for tensor PGD (default: 0.01)
%              .r: Target rank for final matrix
%              .Xstar: Ground truth for error tracking
%              .projection: Projection function for extracted matrix
%              .verbose: Print progress (default: false)
%
% Outputs:
%   X0       - Initialized matrix (d1 x d2) extracted from tensor
%   U0       - Factor matrix from tensor extraction (if available)
%   history  - Struct with convergence information:
%              .tensor_errors: Tensor error at each iteration (if Xstar provided)
%              .loss_function: Loss function value at each iteration
%              .iterations: Number of tensor PGD iterations performed
%              .final_error: Final matrix error after extraction (if Xstar provided)
%              .method: 'tensor_lift'

    %% Validate symmetric case
    if d1 ~= d2
        error('Tensor lift initialization requires symmetric matrices: d1 must equal d2');
    end
    d = d1;
    n = d * d;  % Flattened matrix dimension
    m = length(y);
    
    %% Extract parameters
    if isfield(params, 'T_power')
        T_power = params.T_power;
    else
        T_power = 1; % Default number of tensor iterations
    end
    
    if isfield(params, 'r')
        r = params.r;
    else
        r = 1; % Default rank
    end
    
    verbose = isfield(params, 'verbose') && params.verbose;
    has_ground_truth = isfield(params, 'Xstar') && ~isempty(params.Xstar);
    has_projection = isfield(params, 'projection') && ~isempty(params.projection);
    
    if verbose
        fprintf('--- Tensor Lift Initialization (PGD) ---\n');
        fprintf('Matrix: %dx%d, Rank: %d, Measurements: %d\n', d, d, r, m);
        fprintf('Tensor PGD iterations: %d\n', T_power);
    end
    
    %% Initialize history tracking
    history = struct();
    history.method = 'tensor_lift';
    history.iterations = T_power;
    
    if has_ground_truth
        Xstar = params.Xstar;
        tensor_Xstar = create_tensor_from_matrix(Xstar, d);
    end
    
    %% Lift operator to tensor space
    % Create tensor measurement operators: A_i ⊗ A_i
    if verbose
        fprintf('Lifting operators to tensor space...\n');
    end
    
    A_tensor = zeros(m, n*n);  % Each row is A_i ⊗ A_i flattened
    
    % Extract measurement matrices from operator
    % We need to reconstruct A from the operator structure
    % Assume operator.A is linear: operator.A(X) = A * X(:)
    % We can get A by applying operator.A to standard basis matrices
    A_matrix = zeros(m, n);
    for j = 1:n
        e_j = zeros(n, 1);
        e_j(j) = 1;
        E_j = reshape(e_j, [d, d]);
        A_matrix(:, j) = operator.A(E_j);
    end
    
    % Create tensor operators: A_i ⊗ A_i for each measurement
    for i = 1:m
        Ai = reshape(A_matrix(i, :), [d, d]);
        Ai = (Ai + Ai')/2;  % Symmetrize
        % Fourth-order tensor A_i ⊗ A_i
        AiAi = reshape(Ai, n, 1) * reshape(Ai, 1, n);  % d^2 x d^2
        A_tensor(i, :) = AiAi(:)';  % Flatten and store
    end
    
    % Define tensor operators
    tensor_operator = struct();
    tensor_operator.A = @(T) tensor_forward(T, A_tensor, d);
    tensor_operator.A_star = @(z) tensor_adjoint(z, A_tensor, d);
    
    %% Tensor measurements (same as matrix measurements for phase retrieval)
    y_tensor = y;  % Measurements are the same: |<A_i, X>|
    
    if verbose
        fprintf('Tensor measurements: range=[%.3f, %.3f]\n', min(y_tensor), max(y_tensor));
    end
    
    %% Tensor PGD Initialization
    % Initialize with random tensor
    Xl_tensor_init = zeros(d, d, d, d);
    % Xl_tensor_init = Xl_tensor_init / norm(Xl_tensor_init(:));
    
    % Step size for tensor PGD
    mu = 0.1;
    
    %% Setup Solver Parameters
    solver_params = struct();
    solver_params.T = T_power;
    solver_params.mu = mu;
    solver_params.r = r;
    solver_params.verbose = verbose;
    solver_params.projection = @(X) tensor_projection_rank_r(X, r);
    
    % Add ground truth for error tracking if available
    if has_ground_truth
        solver_params.Xstar = tensor_Xstar;  % Ground truth tensor for error tracking
    end
    
    %% Run Tensor PGD Algorithm
    if verbose
        fprintf('Running tensor PGD initialization (%d iterations)...\n', T_power);
        fprintf('Step size mu=%.4f\n', mu);
    end
    
    % Use solve_PGD to run tensor PGD
    [solver_output, Xl_tensor_final] = solve_PGD(Xl_tensor_init, [], y_tensor, tensor_operator, d, [], [], m, solver_params);
    
    % Store history from solver output
    if has_ground_truth
        history.tensor_errors = solver_output.Error_Stand;
    end
    history.loss_function = solver_output.Error_function;
    
    %% Extract matrix from final tensor
    if verbose
        fprintf('Extracting matrix from tensor...\n');
    end
    
    extract_params = struct();
    extract_params.r = r;
    extract_params.method = 'eig';  % Use eigenvalue extraction
    extract_params.verbose = false;
    
    X0 = extract_matrix_from_tensor(Xl_tensor_final, extract_params);

    %% run some more power iterations (with projection)
    % run 10 steps of power method to refine, based on extracted X0
    solver_params.T_power = 20;
    solver_params.Init = X0(:);
    X0 = initialize_power_method(y, operator, d, d, solver_params);

    
    % Symmetrize (since we're using X = UU^T formulation)
    X0 = (X0 + X0') / 2;
    
    % Apply projection if provided
    if has_projection
        X0 = params.projection(X0);
    end
    
    % Extract factor U0 if possible
    % [U_svd, S_svd, ~] = svd(X0);
    % r_effective = min(r, rank(X0, 1e-10));
    % U0 = U_svd(:, 1:r_effective) * sqrt(S_svd(1:r_effective, 1:r_effective));
    U0 = [];
    % Final error check and compute matrix errors at each iteration
    if has_ground_truth
        [final_error, X0_aligned] = rectify_sign_ambiguity(X0, Xstar);
        X0 = X0_aligned;
        history.final_error = final_error;
        
        % Note: We can't compute matrix_errors at each iteration efficiently here
        % since that would require extracting matrices at each step during solve_PGD
        % The tensor_errors from solve_PGD already give convergence information
        
        if verbose
            fprintf('Final matrix error: %.6e\n', final_error);
            fprintf('Final tensor error: %.6e\n', history.tensor_errors(end));
            fprintf('Final matrix rank: %d (target: %d)\n', rank(X0, 1e-6), r);
        end
    end
    
    if verbose
        fprintf('--- Tensor Lift Initialization Complete ---\n\n');
    end
    
end

%% Helper function
function value = get_param(params, field, default)
    % Get parameter from struct with default value
    if isfield(params, field)
        value = params.(field);
    else
        value = default;
    end
end
