function [output, T_tucker] = solve_RGD_tucker_kronecker(T_tucker, ~, y, tucker_op, ~, ~, ~, m, params)
    % solve_RGD_tucker_kronecker - Riemannian Gradient Descent for Tucker Tensor
    % 
    % This solver minimizes the least-squares loss on Tucker manifold:
    %   ℓ(T) = (1/2m) * ||y - A(T)||²
    % where T is a Tucker tensor: T = G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
    % 
    % Uses Riemannian gradient descent on the Tucker manifold:
    %   1. Forward: y_pred = A(T)
    %   2. Gradient: Compute Riemannian gradient using get_proj_grad_kronecker
    %   3. Retraction: T_{t+1} = Retract(T_t - μ * grad_R)
    %
    % Inputs:
    %   T_tucker - Initial TuckerTensor object (d1 x d2 x d1 x d2)
    %   ~ - Unused (for compatibility)
    %   y - Measurement vector (m x 1)
    %   tucker_op - TuckerOperator object with kronecker forward
    %   ~ - Unused d1 (for compatibility)
    %   ~ - Unused d2 (for compatibility)
    %   ~ - Unused rank (for compatibility)
    %   m - Number of measurements (or can be computed from y)
    %   params - Parameter structure containing:
    %       - T: number of iterations (default: 200)
    %       - mu (or eta): step size (default: 0.01)
    %       - Xstar: ground truth matrix (for error tracking, optional)
    %       - verbose: Print progress (default: false)
    %
    % Outputs:
    %   output - Struct containing auxiliary information:
    %            .Error_Stand    - Standard error tracking (relative error vs Xstar)
    %            .Error_function - Function error tracking (loss values)
    %   T_tucker - Final TuckerTensor solution
    
    %% Validate inputs
    if ~isa(T_tucker, 'TuckerTensor')
        error('First input must be TuckerTensor object');
    end
    
    if ~isa(tucker_op, 'TuckerOperator')
        error('tucker_op must be TuckerOperator object');
    end
    
    %% Extract parameters
    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
    
    if isfield(params, 'mu')
        mu = params.mu;  % Step size
    elseif isfield(params, 'eta')
        mu = params.eta;
    else
        mu = 0.01;  % Default step size
    end
    
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
        has_ground_truth = true;
    else
        has_ground_truth = false;
    end
    
    verbose = get_param(params, 'verbose', false);
    
    % Number of measurements
    if nargin < 8 || isempty(m)
        m = length(y);
    end
    
    %% Initialize error tracking
    Error_Stand = zeros(T, 1);
    Error_function = zeros(T, 1);
    
    %% Compute initial errors
    % Forward pass: compute measurements
    y_pred = tucker_op.forward(T_tucker) / sqrt(m);
    
    % Compute loss (y should already be normalized if measurements are scaled)
    residual = y_pred - y;
    loss = 0.5 * norm(residual)^2;
    Error_function(1) = loss;
    
    if has_ground_truth
        % Extract matrix from Tucker tensor and compute error
        X_current = extract_matrix_from_tucker(T_tucker);
        X_current = (X_current + X_current') / 2;  % Symmetrize
        [Error_Stand(1), ~] = rectify_sign_ambiguity(X_current, Xstar);
    end
    
    if verbose
        fprintf('=== RGD Tucker Kronecker Solver ===\n');
        fprintf('Iterations: %d, Step size: %.4f, Measurements: %d\n', T, mu, m);
        if has_ground_truth
            fprintf('Initial loss: %.6e, Initial error: %.6e\n', loss, Error_Stand(1));
        else
            fprintf('Initial loss: %.6e\n', loss);
        end
        fprintf('-----------------------------------\n');
    end
    
    %% Main RGD iteration loop
    for t = 1:T-1
        % Step 1: Forward pass - compute measurements using kronecker forward
        y_pred = tucker_op.forward(T_tucker) / sqrt(m);
        
        % Step 2: Compute Riemannian gradient on tangent space
        % Using get_proj_grad_kronecker which computes projected gradient
        % Note: following initialize_tensor_lift_tucker.m convention
        Grad_F = tucker_op.get_proj_grad_kronecker(T_tucker, y_pred / sqrt(m), y / sqrt(m));
        
        % Step 3: Retraction - move along tangent direction on manifold
        % T_{t+1} = Retract(T_t - μ * grad_R)
        T_tucker = T_tucker.retraction(Grad_F, mu);
        
        % Step 4: Compute errors for iteration t+1
        y_pred_new = tucker_op.forward(T_tucker) / sqrt(m);
        residual_new = y_pred_new - y;
        loss_new = 0.5 * norm(residual_new)^2;
        Error_function(t+1) = loss_new;
        
        if has_ground_truth
            % Extract matrix from Tucker tensor and compute error
            X_current = extract_matrix_from_tucker(T_tucker);
            X_current = (X_current + X_current') / 2;  % Symmetrize
            [Error_Stand(t+1), ~] = rectify_sign_ambiguity(X_current, Xstar);
        end
        
        % Optional: Print progress
        if verbose && (mod(t, max(1, floor(T/10))) == 0 || t == 1)
            if has_ground_truth
                fprintf('  Iter %4d: Loss = %.6e, Rel. Error = %.6e\n', ...
                    t, loss_new, Error_Stand(t+1));
            else
                fprintf('  Iter %4d: Loss = %.6e\n', t, loss_new);
            end
        end
    end
    
    %% Final output
    if verbose
        fprintf('-----------------------------------\n');
        fprintf('Final loss: %.6e\n', Error_function(end));
        if has_ground_truth
            fprintf('Final relative error: %.6e\n', Error_Stand(end));
        end
        fprintf('=== RGD Tucker Kronecker Complete ===\n');
    end
    
    % Pack output struct
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
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

function X = extract_matrix_from_tucker(T_tucker)
    % EXTRACT_MATRIX_FROM_TUCKER Extract matrix from Tucker tensor
    % For 4th-order tensor T = X ⊗ X, extract X via matricization
    %
    % Method: Matricize T to (d² × d²) and extract leading eigenvector
    
    d = T_tucker.dims(1);
    r = T_tucker.tucker_ranks(1);
    
    % Compute T in matricized form (d² × d²) without forming full tensor
    % T_mat = (U₁ ⊗ U₂) * G_mat * (U₃ ⊗ U₄)'
    % where G_mat is r² × r² matricization of core
    
    U1 = T_tucker.U{1};
    U2 = T_tucker.U{2};
    U3 = T_tucker.U{3};
    U4 = T_tucker.U{4};
    G = T_tucker.G;
    
    % Handle rank-1 case (scalar core)
    if isscalar(G)
        % For scalar core, form Kronecker products directly
        U_left = kron(U1, U2);   % d² × r
        U_right = kron(U3, U4);  % d² × r
        T_mat = G * (U_left * U_right');
    else
        % General case: form matricized core and compute
        G_mat = reshape(permute(G, [1,2,3,4]), [r*r, r*r]);
        
        % Form Kronecker products of factors
        U_left = kron(U1, U2);   % d² × r²
        U_right = kron(U3, U4);  % d² × r²
        
        % Matricized tensor: T_mat = U_left * G_mat * U_right'
        T_mat = U_left * G_mat * U_right';
    end
    
    % Symmetrize
    T_mat = (T_mat + T_mat') / 2;
    
    % Extract leading eigenvector and reshape to matrix
    [V, D] = eig(T_mat);
    [~, idx] = max(abs(diag(D)));
    v_lead = V(:, idx);
    
    % Reshape to matrix
    X = reshape(v_lead, [d, d]);
    
    % Normalize
    X = X / norm(X, 'fro');
    
    % Make symmetric
    X = (X + X') / 2;
end

