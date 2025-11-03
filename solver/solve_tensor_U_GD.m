function [U_final, X_final, history] = solve_tensor_U_GD(y, A_cells, U0, params)
% SOLVE_TENSOR_U_GD Gradient descent on U for tensor-lifted problem
%
% Solves: min_U ||y - A(T)||²
% where T = X ⊗ X, X = UU^T, U ∈ ℝ^(d×r)
%
% This solver performs gradient descent directly on the factor U,
% avoiding explicit Tucker decomposition while still working with
% the tensor formulation T = X ⊗ X.
%
% Inputs:
%   y        - Measurement vector (m × 1)
%   A_cells  - Cell array of measurement matrices {A₁, ..., A_m}
%              where yᵢ = ⟨Aᵢ ⊗ Aᵢ, T⟩
%   U0       - Initial factor matrix (d × r)
%   params   - Struct with fields:
%              .T: Number of iterations (default: 100)
%              .mu: Step size (default: 0.01)
%              .Xstar: Ground truth for error tracking (optional)
%              .verbose: Print progress (default: false)
%
% Outputs:
%   U_final  - Final factor matrix (d × r)
%   X_final  - Final matrix X = U_final * U_final' (d × d)
%   history  - Struct with convergence information
%
% The gradient with respect to U is:
%   ∇_U f = 4 ∑ᵢ (yᵢ - ⟨Aᵢ⊗Aᵢ, X⊗X⟩) · [-(AᵢXAᵢX)U - (AᵢXAᵢX)'U]

    %% Parse parameters
    T = get_param(params, 'T', 100);
    mu = get_param(params, 'mu', 0.01);
    verbose = get_param(params, 'verbose', false);
    has_ground_truth = isfield(params, 'Xstar') && ~isempty(params.Xstar);
    
    if has_ground_truth
        Xstar = params.Xstar;
    end
    
    %% Initialize
    [d, r] = size(U0);
    m = length(y);
    U = U0;
    
    % Initialize history
    history = struct();
    history.loss = zeros(T, 1);
    if has_ground_truth
        history.errors = zeros(T, 1);
    end
    
    if verbose
        fprintf('=== Tensor U Gradient Descent ===\n');
        fprintf('d=%d, r=%d, m=%d, T=%d, mu=%.4f\n', d, r, m, T, mu);
        if has_ground_truth
            fprintf('Iter | Loss      | Error     | Grad Norm\n');
            fprintf('-----|-----------|-----------|----------\n');
        else
            fprintf('Iter | Loss      | Grad Norm\n');
            fprintf('-----|-----------|----------\n');
        end
    end
    
    %% Gradient Descent Loop
    for t = 1:T
        % Current matrix X = UU^T
        X = U * U';
        X = (X + X') / 2;  % Symmetrize
        
        % Forward pass: compute y_pred = A(T) where T = X ⊗ X
        y_pred = forward_tensor_op(X, A_cells)/sqrt(m);
        
        % Compute loss
        residual = y_pred - y;
        loss = 0.5 * norm(residual)^2;
        history.loss(t) = loss;
        
        % Compute gradient ∇_U f
        grad_U = compute_gradient_U(U, X, A_cells, residual)/sqrt(m);
        grad_norm = norm(grad_U, 'fro');
        
        % Gradient descent update
        U = U - mu * grad_U;
        
        % Track error if ground truth available
        if has_ground_truth
            X_current = U * U';
            X_current = (X_current + X_current') / 2;
            [err, ~] = rectify_sign_ambiguity(X_current, Xstar);
            history.errors(t) = err;
            
            if verbose && (mod(t, max(1, floor(T/10))) == 0 || t == 1)
                fprintf('%4d | %.4e | %.4e | %.4e\n', t, loss, err, grad_norm);
            end
        else
            if verbose && (mod(t, max(1, floor(T/10))) == 0 || t == 1)
                fprintf('%4d | %.4e | %.4e\n', t, loss, grad_norm);
            end
        end
    end
    
    %% Finalize
    U_final = U;
    X_final = U * U';
    X_final = (X_final + X_final') / 2;
    
    if verbose
        fprintf('Final loss: %.6e\n', history.loss(end));
        if has_ground_truth
            fprintf('Final error: %.6e\n', history.errors(end));
        end
    end
end

%% Helper Functions

function y_pred = forward_tensor_op(X, A_cells)
    % FORWARD_TENSOR_OP Compute y = A(T) where T = X ⊗ X
    % yᵢ = ⟨Aᵢ ⊗ Aᵢ, X ⊗ X⟩ = trace(AᵢXAᵢX)
    
    m = length(A_cells);
    y_pred = zeros(m, 1);
    
    for i = 1:m
        Ai = A_cells{i};
        % Compute ⟨Aᵢ⊗Aᵢ, X⊗X⟩ = trace(AᵢXAᵢX)
        temp = Ai * X;
        y_pred(i) = trace(temp * temp);
    end
end

function grad_U = compute_gradient_U(U, X, A_cells, residual)
    % COMPUTE_GRADIENT_U Compute gradient ∇_U f
    %
    % The gradient is:
    %   ∇_U f = ∑ᵢ residualᵢ · ∇_U ⟨Aᵢ⊗Aᵢ, X⊗X⟩
    %
    % where ∇_U ⟨Aᵢ⊗Aᵢ, X⊗X⟩ = 4(AᵢXAᵢX)U + 4(AᵢXAᵢX)'U
    %                          = 4[(AᵢXAᵢX) + (AᵢXAᵢX)']U
    %
    % Since X = UU^T is symmetric, we have:
    %   ∂/∂U ⟨Aᵢ⊗Aᵢ, X⊗X⟩ = 4(AᵢXAᵢX)U
    
    [d, r] = size(U);
    m = length(A_cells);
    grad_U = zeros(d, r);
    
    for i = 1:m
        if abs(residual(i)) < 1e-15
            continue;  % Skip if residual is negligible
        end
        
        Ai = A_cells{i};
        
        % Compute AᵢXAᵢX
        AiX = Ai * X;
        AiXAiX = AiX * AiX;
        
        % Symmetrize for numerical stability
        AiXAiX_sym = (AiXAiX + AiXAiX') / 2;
        
        % Gradient contribution: residualᵢ · 4·AiXAiX·U
        grad_U = grad_U + 4 * residual(i) * AiXAiX_sym * U;
    end
end

function value = get_param(params, field, default)
    % Get parameter from struct with default value
    if isfield(params, field)
        value = params.(field);
    else
        value = default;
    end
end
