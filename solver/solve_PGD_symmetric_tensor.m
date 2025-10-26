function [Error_Stand, Error_function, Xl] = solve_PGD_symmetric(Xl, ~, y, operator, d1, d2, ~, m, params)
    % solve_PGD_symmetric - PGD for Symmetric Low-Rank Phase Retrieval
    % 
    % This solver handles the case where X = UU^T is symmetric positive semidefinite.
    % The problem can be viewed as a fourth-order tensor recovery.
    % 
    % Minimizes the amplitude-based loss:
    %   ℓ(X) = (1/2m) * sum_i (y_i - |<A_i, X>|)^2
    % subject to: X = UU^T (symmetric, rank ≤ r)
    %
    % Inputs:
    %   Xl - Initial matrix (d1 x d1, assumed symmetric)
    %   ~ - Unused
    %   y - Measurement vector (magnitudes)
    %   operator - Struct with A and A_star operators
    %   d1, d2 - Matrix dimensions (d1 = d2 for symmetric case)
    %   ~ - Unused
    %   m - Number of measurements
    %   params - Parameters struct with:
    %       - T: iterations
    %       - mu/eta: step size
    %       - r: rank constraint
    %       - Xstar: ground truth (optional)
    %
    % Outputs:
    %   Error_Stand - Relative error over iterations
    %   Error_function - Amplitude loss over iterations
    %   Xl - Final symmetric solution
    
    % Extract parameters
    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
    
    if isfield(params, 'mu')
        eta = params.mu;
    elseif isfield(params, 'eta')
        eta = params.eta;
    else
        eta = 0.2;
    end
    
    if isfield(params, 'r')
        r = params.r;
    else
        r = min(d1, 10);
    end
    
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
        has_ground_truth = true;
    else
        has_ground_truth = false;
    end
    
    if nargin < 8 || isempty(m)
        m = length(y);
    end
    
    % Initialize tracking
    Error_Stand = zeros(T, 1);
    Error_function = zeros(T, 1);
    
    % Ensure initial X is symmetric
    Xl = (Xl + Xl') / 2;
    
    % Compute initial errors
    z = operator.A(Xl);
    amplitude_residual = y - abs(z);
    Error_function(1) = (1/(2*m)) * norm(amplitude_residual)^2;
    
    if has_ground_truth
        [Error_Stand(1), ~] = rectify_sign_ambiguity(Xl, Xstar);
    end
    
    fprintf('PGD-Symmetric: Initial loss = %.6e\n', Error_function(1));
    
    % PGD iteration loop
    for t = 1:T-1
        % Step 1: Compute gradient of amplitude loss
        z = operator.A(Xl);
        abs_z = abs(z);
        amplitude_residual = y - abs_z;
        
        epsilon = 1e-12;
        safe_abs_z = abs_z + epsilon;
        
        % Gradient coefficient
        grad_coeff = amplitude_residual .* conj(z) ./ (safe_abs_z.^2);
        
        % Apply adjoint operator
        gradient = operator.A_star(grad_coeff);
        gradient = -(1/m) * gradient;
        
        if isvector(gradient)
            gradient = reshape(gradient, [d1, d2]);
        end
        
        % Step 2: Symmetrize gradient (enforce symmetric constraint)
        gradient = (gradient + gradient') / 2;
        
        % Step 3: Gradient descent step
        Xl_temp = Xl - eta * gradient;
        
        % Step 4: Enforce symmetry
        Xl_temp = (Xl_temp + Xl_temp') / 2;
        
        % Step 5: Project onto symmetric rank-r constraint
        % For symmetric matrix: X = UU^T where U is d1 x r
        Xl = project_symmetric_low_rank(Xl_temp, r);
        
        % Step 6: Compute errors
        z_new = operator.A(Xl);
        amplitude_residual_new = y - abs(z_new);
        Error_function(t+1) = (1/(2*m)) * norm(amplitude_residual_new)^2;
        
        if has_ground_truth
            [Error_Stand(t+1), ~] = rectify_sign_ambiguity(Xl, Xstar);
        end
        
        % Print progress
        if mod(t, 50) == 0 || t == 1
            if has_ground_truth
                fprintf('  Iter %4d: Loss = %.6e, Rel. Error = %.6e, Rank = %d\n', ...
                    t, Error_function(t+1), Error_Stand(t+1), rank(Xl, 1e-6));
            else
                fprintf('  Iter %4d: Loss = %.6e, Rank = %d\n', ...
                    t, Error_function(t+1), rank(Xl, 1e-6));
            end
        end
    end
    
    fprintf('PGD-Symmetric: Final loss = %.6e\n', Error_function(end));
    if has_ground_truth
        fprintf('PGD-Symmetric: Final relative error = %.6e\n', Error_Stand(end));
    end
    fprintf('PGD-Symmetric: Final rank = %d (constraint: %d)\n', rank(Xl, 1e-6), r);
end

%% Helper Function: Symmetric Low-Rank Projection
function X_proj = project_symmetric_low_rank(X, r)
    % Project onto symmetric rank-r matrices using eigendecomposition
    % For symmetric X, we use eigen-decomposition instead of SVD
    % Result: X = U * Lambda * U' where Lambda has at most r non-zero eigenvalues
    
    % Ensure symmetry
    X = (X + X') / 2;
    
    if r <= 0 || r >= size(X, 1)
        X_proj = X;
        return;
    end
    
    % Eigen-decomposition (symmetric matrix)
    [V, D] = eig(X);
    
    % Sort eigenvalues in descending order by absolute value
    [~, idx] = sort(abs(diag(D)), 'descend');
    V = V(:, idx);
    D = D(idx, idx);
    
    % Keep only top r eigenvalues
    D_proj = zeros(size(D));
    D_proj(1:r, 1:r) = D(1:r, 1:r);
    
    % Reconstruct: X = V * D_proj * V'
    X_proj = V * D_proj * V';
    
    % Ensure exact symmetry (numerical stability)
    X_proj = (X_proj + X_proj') / 2;
end
