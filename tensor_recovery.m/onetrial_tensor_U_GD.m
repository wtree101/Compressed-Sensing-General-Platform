function [err_hist] = onetrial_tensor_U_GD(trial_params)
% ONETRIAL_TENSOR_U_GD Single trial for tensor U gradient descent
%
% Wrapper function for solve_tensor_U_GD that fits the onetrial interface
% used in phase diagram experiments.
%
% This solver performs gradient descent directly on U ∈ ℝ^(d×r) where:
%   X = UU^T, T = X ⊗ X
%   Loss: ||y - A(T)||²
%
% Input:
%   trial_params - Struct with fields:
%                  .d1, .d2: Matrix dimensions (must be equal)
%                  .m: Number of measurements
%                  .r: Rank
%                  .T: Number of iterations
%                  .Xstar: Ground truth matrix
%                  .verbose: Verbosity level
%                  .init: Initialization function handle (optional)
%                  .nonlinear_func: Measurement function (optional)
%
% Output:
%   err_hist - Error history over iterations (T × 1)

    %% Extract parameters
    d1 = trial_params.d1;
    d2 = trial_params.d2;
    m = trial_params.m;
    r = trial_params.r;
    T = trial_params.T;
    Xstar = trial_params.Xstar;
    verbose = trial_params.verbose;
    
    if d1 ~= d2
        error('Tensor U GD requires square matrices: d1 must equal d2');
    end
    d = d1;
    
    %% Generate measurement matrices
    A_cells = cell(m, 1);
    y = zeros(m, 1);
    
    for i = 1:m
        % Generate random Gaussian measurement matrix
        Ai = randn(d, d);
        Ai = (Ai + Ai') / 2;  % Symmetrize
        A_cells{i} = Ai;
        
        % Compute measurement: yᵢ = ⟨Aᵢ⊗Aᵢ, Xstar⊗Xstar⟩ = trace(AᵢXstarAᵢXstar)
        temp = Ai * Xstar;
        y(i) = trace(temp * temp);
    end
    
    % Apply nonlinear function if specified (e.g., for phase retrieval)
    if isfield(trial_params, 'nonlinear_func') && ~isempty(trial_params.nonlinear_func)
        y = trial_params.nonlinear_func(y);
    end
    
    % Apply preprocessing function if specified
    if isfield(trial_params, 'pre_func') && ~isempty(trial_params.pre_func)
        y = trial_params.pre_func(y);
    end
    
    %% Initialization
    if isfield(trial_params, 'init') && ~isempty(trial_params.init)
        % Use provided initialization function
        init_func = trial_params.init;
        
        % Create operator struct for initialization
        operator = struct();
        A_matrix = zeros(m, d*d);
        for i = 1:m
            A_matrix(i, :) = A_cells{i}(:)';
        end
        operator.A = @(X) forward_op_matrix(X, A_matrix, d);
        
        % Call initialization
        init_params = struct();
        init_params.T_power = get_param(trial_params, 'T_power', 10);
        init_params.mu = 0.01;
        init_params.r = r;
        init_params.verbose = 0;
        
        try
            [X0, ~, ~] = init_func(y, operator, d, d, init_params);
            
            % Extract U0 from X0 via eigendecomposition
            [V, D] = eig(X0);
            [~, idx] = sort(diag(D), 'descend');
            V = V(:, idx);
            D = D(idx, idx);
            
            % Take top r eigenvectors
            U0 = V(:, 1:r) * sqrt(max(D(1:r, 1:r), 0));
        catch
            warning('Initialization failed, using random initialization');
            U0 = randn(d, r) * 0.01;
        end
    else
        % Random initialization
        U0 = randn(d, r) * 0.01;
    end
    
    %% Run gradient descent
    solver_params = struct();
    solver_params.T = T;
    solver_params.mu = get_param(trial_params, 'mu', 0.01);
    solver_params.Xstar = Xstar;
    solver_params.verbose = verbose;
    
    [U_final, X_final, history] = solve_tensor_U_GD(y, A_cells, U0, solver_params);
    
    %% Extract error history
    if isfield(history, 'errors')
        err_hist = history.errors;
    else
        % Compute errors post-hoc if not tracked during optimization
        err_hist = zeros(T, 1);
        warning('Error history not available, computing post-hoc...');
        [err_hist(end), ~] = rectify_sign_ambiguity(X_final, Xstar);
        % Fill with final error (not accurate but maintains interface)
        err_hist(:) = err_hist(end);
    end
end

%% Helper Functions

function y = forward_op_matrix(X, A_matrix, d)
    % Forward operator using matrix format
    m = size(A_matrix, 1);
    y = zeros(m, 1);
    for i = 1:m
        Ai = reshape(A_matrix(i, :), [d, d]);
        Ai = (Ai + Ai') / 2;
        temp = Ai * X;
        y(i) = trace(temp * temp);
    end
end

function value = get_param(params, field, default)
    if isfield(params, field)
        value = params.(field);
    else
        value = default;
    end
end
