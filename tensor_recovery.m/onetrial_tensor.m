function [Error_Stand, Error_function, is_success] = onetrial_tensor(params)
% ONETRIAL_TENSOR Single trial of symmetric fourth-order tensor phase retrieval
% 
% This function performs one trial of tensor phase retrieval using the 
% formulation X = UU^T viewed as fourth-order tensor T = X ⊗ X
% Linear model: y_i = ⟨A_i ⊗ A_i, T⟩
%
% Inputs:
%   params - Structure with required and optional fields:
%            Required: m, r, d (or d1=d2=d)
%            Optional: kappa, Xstar, verbose, T, mu, init_method, etc.
%            
% Outputs:
%   Error_Stand     - Relative error history vs ground truth
%   Error_function  - Loss function history  
%   Xl_tensor_final - Final recovered tensor

    %% Parameter Validation and Setup
    % Required parameters
    if ~isfield(params, 'm'), error('Parameter m (measurements) is required'); end
    if ~isfield(params, 'r'), error('Parameter r (rank) is required'); end
    
    % Handle symmetric case: d1 = d2 = d
    if isfield(params, 'd')
        d = params.d;
    elseif isfield(params, 'd1') && isfield(params, 'd2')
        d1 = params.d1; d2 = params.d2;
        if d1 ~= d2
            error('Tensor formulation requires symmetric matrices: d1 must equal d2');
        end
        d = d1;
    else
        error('Either params.d or both params.d1 and params.d2 must be specified');
    end
    
    % Extract core parameters
    m = params.m;
    r = params.r;
    n = d * d;  % Flattened matrix dimension
    
    % Optional parameters with defaults
    verbose = get_param(params, 'verbose', 0);
    T = get_param(params, 'T', 100);  % Number of iterations
    mu = get_param(params, 'mu', 0.01);  % Step size
    init_method = get_param(params, 'init_method', 'zero');
    
    if verbose
        fprintf('=== Tensor Phase Retrieval Trial ===\n');
        fprintf('Configuration: %dx%d matrix, rank=%d, m=%d measurements\n', d, d, r, m);
    end
    
    %% Generate or Use Ground Truth
    if isfield(params, 'Xstar') && ~isempty(params.Xstar)
        Xstar = params.Xstar;
        if verbose
            fprintf('Using provided ground truth matrix\n');
        end
    else
        % Generate symmetric low-rank ground truth: X = UU^T
        if verbose
            fprintf('Generating symmetric ground truth: X = UU^T\n');
        end
        U_true = randn(d, r);
        Xstar = U_true * U_true';  % Symmetric rank-r matrix
        Xstar = Xstar / norm(Xstar, 'fro');  % Normalize
    end
    
    % Create fourth-order tensor: T = X ⊗ X
    tensor_Xstar = create_tensor_from_matrix(Xstar, d);
    
    if verbose
        fprintf('Ground truth: rank=%d, norm=%.6f, symmetry_error=%.2e\n', ...
                rank(Xstar), norm(Xstar, 'fro'), norm(Xstar - Xstar', 'fro'));
    end
    
    %% Generate Tensor Measurements
    if verbose
        fprintf('Generating tensor measurements A_i ⊗ A_i...\n');
    end
    
    % Create measurement operators for fourth-order tensors
    A_tensor = zeros(m, n*n);  % Each row is A_i ⊗ A_i flattened
    for i = 1:m
        Ai = randn(d, d);
        Ai = (Ai + Ai')/2;  % Symmetric measurement matrices
        % Fourth-order tensor A_i ⊗ A_i
        AiAi = reshape(Ai, n, 1) * reshape(Ai, 1, n);  % d^2 x d^2
        A_tensor(i, :) = AiAi(:)';  % Flatten and store
    end
    
    % Define tensor operators
    operator = struct();
    operator.A = @(T) tensor_forward(T, A_tensor, d);
    operator.A_star = @(z) tensor_adjoint(z, A_tensor, d);
    
    % Generate measurements
    y = operator.A(tensor_Xstar) / sqrt(m);
    
    % Apply nonlinear transformation if specified
    if isfield(params, 'nonlinear_func') && ~isempty(params.nonlinear_func)
        y = params.nonlinear_func(y);
        if verbose
            fprintf('Applied nonlinear transformation\n');
        end
    end
    
    if verbose
        fprintf('Measurements: range=[%.3f, %.3f], mean=%.3f\n', min(y), max(y), mean(y));
    end
    
    %% Initialization
    switch lower(init_method)
        case 'zero'
            if verbose, fprintf('Using zero initialization\n'); end
            Xl_tensor_init = zeros(d, d, d, d);
            
        case 'random'
            if verbose, fprintf('Using random initialization\n'); end
            Xl_tensor_init = randn(d, d, d, d) * 0.01;
            % Apply tensor projection to ensure proper structure
            proj_params = struct('r', r);
            Xl_tensor_init = tensor_projection_rank_r(Xl_tensor_init, proj_params);
            
        case 'power'
            if verbose, fprintf('Using power method initialization\n'); end
            % Use power method initialization adapted for tensors
            init_params = struct();
            init_params.is_matrix = false;  % Tensor mode
            init_params.r = r;
            init_params.Xstar = tensor_Xstar;
            if isfield(params, 'prefunc')
                init_params.prefunc = params.prefunc;
            end
            
            % Initialize with power method, then convert to tensor
            [Xl_init_matrix, ~] = initialize_power_method(y, operator, d, d, 50, init_params);
            Xl_tensor_init = create_tensor_from_matrix(Xl_init_matrix, d);
            
        otherwise
            error('Unknown initialization method: %s', init_method);
    end
    
    %% Setup Solver Parameters
    solver_params = struct();
    solver_params.T = T;
    solver_params.Xstar = tensor_Xstar;  % Ground truth tensor for error tracking
    solver_params.mu = mu;
    solver_params.r = r;
    solver_params.verbose = verbose;
    solver_params.projection = @tensor_projection_rank_r;
    
    % Copy additional solver parameters from input
    solver_fields = {'tol', 'max_iter', 'adaptive_step', 'projection_freq'};
    for i = 1:length(solver_fields)
        field = solver_fields{i};
        if isfield(params, field)
            solver_params.(field) = params.(field);
        end
    end
    
    %% Run Tensor PGD Algorithm
    if verbose
        fprintf('Running tensor PGD algorithm...\n');
    end
    
    if isfield(params, 'alg') && ~isempty(params.alg)
        % Use specified solver
        [Error_Stand, Error_function, Xl_tensor_final] = params.alg(Xl_tensor_init, [], y, operator, d, [], [], m, solver_params);
    else
        % Use default tensor PGD solver
        [Error_Stand, Error_function, Xl_tensor_final] = solve_PGD_tensor(Xl_tensor_init, [], y, operator, d, [], [], m, solver_params);
    end
    
    %% Final Analysis
    if verbose
        % Extract matrix from final tensor for analysis
        extract_params = struct('r', r, 'method', 'eig', 'verbose', false);
        Xl_final = extract_matrix_from_tensor(Xl_tensor_final, extract_params);
        
        % Compute final error with sign rectification
        [final_error, ~] = rectify_sign_ambiguity(Xl_final, Xstar);
        if final_error < 1e-2
            is_success = true;
        else
            is_success = false;
        end
        
        fprintf('Final results:\n');
        fprintf('  Relative error: %.6e\n', final_error);
        fprintf('  Final loss: %.6e\n', Error_function(end));
        fprintf('  Recovered rank: %d (true: %d)\n', rank(Xl_final, 1e-6), r);
        fprintf('  Iterations completed: %d\n', length(Error_Stand));
        
        % Convergence plot if requested
        if verbose >= 2
            figure;
            semilogy(1:length(Error_Stand), Error_Stand, 'b-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Relative Error');
            title('Tensor Phase Retrieval Convergence');
            grid on;
        end
    end

end

%% Helper Functions
function val = get_param(params, field, default_val)
    % Get parameter value with default fallback
    if isfield(params, field)
        val = params.(field);
    else
        val = default_val;
    end
end

function tensor_T = create_tensor_from_matrix(X, d)
    % Create fourth-order tensor T = X ⊗ X from matrix X
    n = d * d;
    X_vec = reshape(X, n, 1);
    tensor_flat = X_vec * X_vec';  % n x n matrix
    tensor_T = reshape(tensor_flat, [d, d, d, d]);  % 4D tensor
end

function y = tensor_forward(T, A_tensor, d)
    % Forward operator: y = A(T) for 4D tensor T
    n = d * d;
    T_vec = reshape(T, [n * n, 1]);
    y = A_tensor * T_vec;
end

function T = tensor_adjoint(z, A_tensor, d)
    % Adjoint operator: T = A^*(z) returns 4D tensor
    T_vec = A_tensor' * z;
    T = reshape(T_vec, [d, d, d, d]);
end

