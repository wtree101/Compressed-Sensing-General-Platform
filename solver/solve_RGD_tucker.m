function [output, T_final] = solve_RGD_tucker(T_init, y, tucker_op, params)
% SOLVE_RGD_TUCKER Riemannian Gradient Descent for Tucker tensor recovery
%
% Solves: min_{T} (1/2m)||y - A(T)||²
% where T is a Tucker tensor: T = G ×₁ U₁ ×₂ U₂ ×₃ ... ×_N U_N
%
% Uses Riemannian gradient descent on the Tucker manifold:
%   1. Compute Euclidean gradient: ∇f = A*(A(T) - y)
%   2. Project to tangent space: grad_R = P_T(∇f)
%   3. Retract: T_{k+1} = R_T(T_k - μ * grad_R)
%
% Inputs:
%   T_init    - Initial TuckerTensor object
%   y         - Measurement vector (m × 1)
%   tucker_op - TuckerOperator object
%   params    - Struct with fields:
%               .T: Number of iterations
%               .mu: Step size
%               .Xstar: Ground truth (optional, for error tracking)
%               .verbose: Print progress (default: 0)
%
% Outputs:
%   output  - Struct with .Error_Stand, .Error_function
%   T_final - Final TuckerTensor object

    %% Extract parameters
    T_max = params.T;
    mu = params.mu;
    verbose = get_param(params, 'verbose', 0);
    has_ground_truth = isfield(params, 'Xstar') && ~isempty(params.Xstar);
    
    if has_ground_truth
        Xstar = params.Xstar;
    end
    
    m = length(y);
    
    %% Validate inputs
    if ~isa(T_init, 'TuckerTensor')
        error('T_init must be TuckerTensor object');
    end
    if ~isa(tucker_op, 'TuckerOperator')
        error('tucker_op must be TuckerOperator object');
    end
    
    %% Initialize tracking
    Error_Stand = zeros(T_max, 1);
    Error_function = zeros(T_max, 1);
    
    %% Copy initial tensor
    T_current = T_init;
    
    if verbose
        fprintf('=== Riemannian GD for Tucker Tensor ===\n');
        fprintf('Iterations: %d, Step size: %.4f\n', T_max, mu);
        T_current.display();
    end
    
    %% Main RGD loop
    for iter = 1:T_max
        %% 1. Forward pass: compute A(T)
        y_pred = tucker_op.forward(T_current);
        
        %% 2. Compute residual and loss
        residual = y_pred / sqrt(m) - y;
        loss = 0.5 * norm(residual)^2;
        Error_function(iter) = loss;
        
        %% 3. Compute gradients via adjoint: A*(residual)
        [grad_G, grad_U] = tucker_op.adjoint(residual / sqrt(m), T_current);
        
        %% 4. Project gradient to tangent space (Riemannian manifold)
        if T_current.is_symmetric
            % Single factor: project to orthogonal complement
            U = T_current.U{1};
            d = size(U, 1);
            grad_U_tangent = (eye(d) - U * U') * grad_U;
        else
            % Multiple factors: project each independently
            grad_U_tangent = cell(size(grad_U));
            for k = 1:T_current.order
                Uk = T_current.U{k};
                dk = size(Uk, 1);
                grad_U_tangent{k} = (eye(dk) - Uk * Uk') * grad_U{k};
            end
        end
        
        %% 5. Gradient descent update
        G_new = T_current.G - mu * grad_G;
        
        if T_current.is_symmetric
            U_new = T_current.U{1} - mu * grad_U_tangent;
            % Retraction: QR to maintain orthogonality
            [U_new, ~] = qr(U_new, 0);
        else
            U_new = cell(size(T_current.U));
            for k = 1:T_current.order
                U_new{k} = T_current.U{k} - mu * grad_U_tangent{k};
                [U_new{k}, ~] = qr(U_new{k}, 0);
            end
        end
        
        %% 6. Update tensor
        T_current = T_current.update_G(G_new);
        if T_current.is_symmetric
            T_current = T_current.update_U(1, U_new);
        else
            for k = 1:T_current.order
                T_current = T_current.update_U(k, U_new{k});
            end
        end
        
        %% 7. Track error vs ground truth (if available)
        if has_ground_truth && (mod(iter, 10) == 0 || iter == T_max)
            % Extract matrix from Tucker tensor for comparison
            X_current = extract_matrix_from_tucker_tensor(T_current, params.r);
            [err, ~] = rectify_sign_ambiguity(X_current, Xstar);
            Error_Stand(iter) = err;
            
            if verbose
                fprintf('Iter %3d: Loss=%.4e, RelError=%.4e\n', ...
                        iter, loss, err);
            end
        elseif verbose && (mod(iter, 10) == 0 || iter == 1)
            fprintf('Iter %3d: Loss=%.4e\n', iter, loss);
        end
    end
    
    %% Pack output
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
    
    T_final = T_current;
    
    if verbose
        fprintf('=== Tucker-RGD Complete ===\n');
        fprintf('Final loss: %.4e\n', Error_function(end));
        if has_ground_truth
            fprintf('Final error: %.4e\n', Error_Stand(end));
        end
    end
end

function X = extract_matrix_from_tucker_tensor(T_tucker, r)
    % Extract matrix X from Tucker tensor T = X ⊗ X
    % For symmetric 4th-order tensor representing X ⊗ X
    
    if T_tucker.order ~= 4
        error('Matrix extraction requires 4th-order tensor');
    end
    
    G = T_tucker.G;
    U = T_tucker.U{1};
    [d, r_tucker] = size(U);
    
    % Build matrix M = U * G_sum * U' where G_sum aggregates over modes 3,4
    M = zeros(d, d);
    for i3 = 1:r_tucker
        for i4 = 1:r_tucker
            G_slice = G(:, :, i3, i4);  % (r_tucker × r_tucker)
            M = M + U * G_slice * U';
        end
    end
    
    % Symmetrize
    M = (M + M') / 2;
    
    % Extract leading eigenvectors
    [V, D] = eigs(M, min(r, size(M,1)), 'largestabs');
    
    % Form rank-r approximation
    X = V * sqrt(abs(D)) * V';
    
    % Normalize
    X = X / norm(X, 'fro');
end

function value = get_param(params, field, default)
    if isfield(params, field)
        value = params.(field);
    else
        value = default;
    end
end
