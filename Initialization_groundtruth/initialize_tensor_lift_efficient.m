function [X0, U0, history] = initialize_tensor_lift_efficient(y, operator, d1, d2, params)
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
    debug_mode = isfield(params, 'debug') && params.debug;
    
    if verbose
        fprintf('--- Tensor Lift Initialization (PGD) ---\n');
        fprintf('Matrix: %dx%d, Rank: %d, Measurements: %d\n', d, d, r, m);
        fprintf('Tensor PGD iterations: %d\n', T_power);
        if debug_mode
            fprintf('DEBUG MODE: ON\n');
        end
    end
    
    %% Initialize history tracking
    history = struct();
    history.method = 'tensor_lift';
    history.iterations = T_power;
    
    if has_ground_truth
        Xstar = params.Xstar;
        tensor_Xstar = create_tensor_from_matrix(Xstar, d);
        
        if debug_mode
            fprintf('\n=== DEBUG: Ground Truth Tensor ===\n');
            fprintf('Xstar dimensions: [%s]\n', num2str(size(Xstar)));
            fprintf('Xstar norm: %.6f\n', norm(Xstar, 'fro'));
            fprintf('Xstar rank: %d\n', rank(Xstar, 1e-10));
            fprintf('Tensor_Xstar dimensions: [%s]\n', num2str(size(tensor_Xstar)));
            fprintf('Tensor_Xstar norm: %.6f\n', norm(tensor_Xstar(:)));
            
            % Verify tensor is correctly formed: T = X ⊗ X
            % Check a few entries
            fprintf('Verification: T(i,j,k,l) = X(i,j) * X(k,l)\n');
            for test_iter = 1:3
                i = randi(d); j = randi(d); k = randi(d); l = randi(d);
                tensor_val = tensor_Xstar(i, j, k, l);
                matrix_val = Xstar(i, j) * Xstar(k, l);
                diff = abs(tensor_val - matrix_val);
                fprintf('  T(%d,%d,%d,%d) = %.6f, X(%d,%d)*X(%d,%d) = %.6f, diff = %.2e\n', ...
                        i, j, k, l, tensor_val, i, j, k, l, matrix_val, diff);
            end
            fprintf('\n');
        end
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
    
    %% DEBUG: Compute tensor loss for ground truth
    if debug_mode && has_ground_truth
        fprintf('=== DEBUG: Ground Truth Tensor Loss ===\n');
        
        % Compute forward on ground truth tensor
        y_star = tensor_operator.A(tensor_Xstar);
        
        % Compute loss: ||y - A(T_star)||^2
        residual_star = y_tensor - y_star;
        loss_star = 0.5 * norm(residual_star)^2;
        rel_residual_star = norm(residual_star) / norm(y_tensor);
        
        fprintf('Ground truth tensor forward output: range=[%.3f, %.3f]\n', ...
                min(y_star), max(y_star));
        fprintf('Measurements y: range=[%.3f, %.3f], norm=%.6f\n', ...
                min(y_tensor), max(y_tensor), norm(y_tensor));
        fprintf('Residual: norm=%.6f (%.2f%% of ||y||)\n', ...
                norm(residual_star), 100*rel_residual_star);
        fprintf('Ground truth loss: %.6e\n', loss_star);
        
        % Check if measurements are consistent with ground truth
        if rel_residual_star < 1e-10
            fprintf('✓ Measurements are consistent with ground truth tensor\n');
        else
            fprintf('⚠ Measurements have noise or inconsistency (rel residual=%.2e)\n', ...
                    rel_residual_star);
        end
        
        % Verify adjoint: <A(T), y> = <T, A*(y)>
        adj_test = tensor_operator.A_star(y_tensor);
        fprintf('Adjoint test dimensions: [%s]\n', num2str(size(adj_test)));
        inner1 = sum(y_star(:) .* y_tensor(:));
        inner2 = sum(tensor_Xstar(:) .* adj_test(:));
        fprintf('Adjoint verification: <A(T),y>=%.6f, <T,A*(y)>=%.6f, diff=%.2e\n', ...
                inner1, inner2, abs(inner1-inner2));
        fprintf('\n');
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

function T = create_tensor_from_matrix(X, d)
    % CREATE_TENSOR_FROM_MATRIX Create 4th-order tensor T = X ⊗ X
    % 
    % For matrix X of size (d x d), creates tensor T of size (d x d x d x d)
    % such that T(i,j,k,l) = X(i,j) * X(k,l)
    %
    % Input:
    %   X - Matrix of size (d x d)
    %   d - Dimension
    %
    % Output:
    %   T - Fourth-order tensor of size (d x d x d x d)
    
    % Vectorize for efficiency
    X_vec = X(:);  % d^2 x 1
    T_mat = X_vec * X_vec';  % d^2 x d^2 (outer product)
    T = reshape(T_mat, [d, d, d, d]);  % Reshape to 4th-order tensor
end
