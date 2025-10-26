function [output, is_success] = onetrial_MatTensor_PGD(params)
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
%   output      - Struct with auxiliary information:
%                 .Error_Stand    - Relative error history vs ground truth
%                 .Error_function - Loss function history
%                 .Xl_tensor_final - Final recovered tensor
%   is_success  - Binary flag: 1 if recovered successfully, 0 otherwise

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
    % Use the initialization function handle (similar to onetrial.m)
    if isfield(params, 'init') && ~isempty(params.init)
        % Use the function handle from set_init
        init_flag = get_param(params, 'init_flag', 1);
        if init_flag == 0
            [Xl_tensor_init, ~] = params.init(y, operator, d, d, r, m);
        else
            init_scale = get_param(params, 'init_scale', 1.0);
            [Xl_tensor_init, ~] = params.init(y, operator, d, d, r, m, init_scale);
        end
    else
        % Fallback to zero initialization if no init function specified
        if verbose, fprintf('No init function specified, using zero initialization\n'); end
        Xl_tensor_init = zeros(d, d, d, d);
    end
    
    %% Setup Solver Parameters
    solver_params = struct();
    solver_params.T = T;
    solver_params.Xstar = tensor_Xstar;  % Ground truth tensor for error tracking
    solver_params.mu = mu;
    solver_params.r = r;
    solver_params.verbose = verbose;
    solver_params.projection = @(X) tensor_projection_rank_r(X,r);
    
   
    
    %% Run Tensor PGD Algorithm
    if verbose
        fprintf('Running tensor PGD algorithm...\n');
    end
    
    if isfield(params, 'alg_func') && ~isempty(params.alg_func)
        % Use specified solver
        [solver_output, Xl_tensor_final] = params.alg_func(Xl_tensor_init, [], y, operator, d, [], [], m, solver_params);
        Error_Stand = solver_output.Error_Stand;
        Error_function = solver_output.Error_function;
    else
        % Use default tensor PGD solver
        [solver_output, Xl_tensor_final] = solve_PGD(Xl_tensor_init, [], y, operator, d, [], [], m, solver_params);
        Error_Stand = solver_output.Error_Stand;
        Error_function = solver_output.Error_function;
    end
    %% Refine using PGD
    
    %% Final Analysis and Success Check
    % Always extract matrix and compute success (needed for multipletrial)
    extract_params = struct('r', r, 'method', 'eig', 'verbose', false);
    Xl_final = extract_matrix_from_tensor(Xl_tensor_final, extract_params);
    
    % Compute final error with sign rectification
    [final_error, ~] = rectify_sign_ambiguity(Xl_final, Xstar);
    
    % Check success criterion
    if final_error < 1e-2
        is_success = 1;
    else
        is_success = 0;
    end
    
    % Pack output struct
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
    output.Xl_tensor_final = Xl_tensor_final;
    output.final_error = final_error;
    output.recovered_rank = rank(Xl_final, 1e-6);
    
    if verbose
        fprintf('Final results:\n');
        fprintf('  Relative error: %.6e\n', final_error);
        fprintf('  Final loss: %.6e\n', Error_function(end));
        fprintf('  Recovered rank: %d (true: %d)\n', output.recovered_rank, r);
        fprintf('  Iterations completed: %d\n', length(Error_Stand));
        fprintf('  Success: %s\n', iif(is_success, 'YES', 'NO'));
        
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
function result = iif(condition, true_val, false_val)
    % Inline if function (kept local for simplicity)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

