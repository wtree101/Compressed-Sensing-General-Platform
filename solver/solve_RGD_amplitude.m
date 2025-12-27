function [output, Xl] = solve_RGD_amplitude(Xl, ~, y, operator, d1, d2, ~, m, params)
    % solve_RGD_amplitude - Preconditioned Riemannian Gradient Descent for Amplitude-Based Loss
    % 
    % This solver minimizes the amplitude-based loss using Riemannian optimization
    % on the fixed-rank manifold with efficient SVD factorization storage.
    %
    % Problem:
    %   min_X  ℓ(X) = (1/2m) * sum_i (y_i - |<A_i, X>|)^2
    %   s.t.   rank(X) ≤ r
    % 
    % Algorithm: Preconditioned Riemannian Gradient Descent (PRGD)
    %   Initialize: X_0 = H_r(A^* y) = U_0 Σ_0 V_0^T
    %   While not converged:
    %     1. Compute gradient: G_t = A^*(A(X_t) - y) (for amplitude loss)
    %     2. Update preconditioners (optional):
    %        L_t = ε_t I + diag(G_t G_t^T)
    %        R_t = ε_t I + diag(G_t^T G_t)
    %        Ũ_t = U_t (U_t^T L_t^{1/4} U_t)^{-1/2}
    %        Ṽ_t = V_t (V_t^T R_t^{1/4} V_t)^{-1/2}
    %     3. Compute search direction and project to tangent space:
    %        D_t = P̃_T(L_t^{-1/4} G_t R_t^{-1/4})
    %        W_t = X_t - α_t * D_t
    %     4. Efficient rank-r truncation: X_{t+1} = H_r(W_t)
    %        Using QR factorization to avoid full SVD
    %
    % Memory Efficiency:
    %   - Stores X_t as U_t (n1×r), Σ_t (r×r), V_t (n2×r) instead of full n1×n2 matrix
    %   - All operations performed on factorized form
    %   - Rank-r truncation via small 2r×2r SVD
    %
    % Inputs:
    %   Xl - Initial matrix (d1 x d2), can be empty for automatic initialization
    %   ~ - Unused (for compatibility)
    %   y - Measurement vector (magnitudes)
    %   operator - Struct containing A (forward) and A_star (adjoint) operators
    %   d1, d2 - Matrix dimensions
    %   ~ - Unused rank parameter (now in params)
    %   m - Number of measurements (or can be computed from y)
    %   params - Parameter structure containing:
    %       - T: number of iterations
    %       - mu (or eta or alpha): step size
    %       - r (or rank): target rank for projection
    %       - use_preconditioner: enable/disable preconditioning (default: true)
    %       - epsilon: regularization for preconditioner (default: 1e-8)
    %       - Xstar: ground truth (for error tracking)
    %       - use_spectral_init: use spectral initialization (default: true)
    %       - verbose: verbosity level (default: 0)
    %       - return_factorized: return U, Sigma, V instead of full matrix (default: false)
    %
    % Outputs:
    %   output - Struct containing auxiliary information:
    %            .Error_Stand    - Standard error tracking
    %            .Error_function - Function error tracking
    %            .U, .Sigma, .V  - SVD factors (if return_factorized=true)
    %   Xl - Final solution (full matrix or struct with .U, .Sigma, .V)
    
    % Extract parameters
    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
    
    if isfield(params, 'mu')
        alpha = params.mu;  % Step size
    elseif isfield(params, 'alpha')
        alpha = params.alpha;
    elseif isfield(params, 'eta')
        alpha = params.eta;
    else
        alpha = 0.1;  % Default step size
    end
    
    % Rank parameter
    if isfield(params, 'r')
        r = params.r;
    elseif isfield(params, 'rank')
        r = params.rank;
    else
        r = min(5, min(d1, d2));  % Default rank
    end
    
    % Preconditioner settings
    if isfield(params, 'use_preconditioner')
        use_preconditioner = params.use_preconditioner;
    else
        use_preconditioner = true;  % Default: use preconditioner
    end
    
    if isfield(params, 'epsilon')
        epsilon_reg = params.epsilon;
    else
        epsilon_reg = 1e-8;  % Default regularization
    end
    
    % Preconditioner power setting (1/4 or 1/2)
    if isfield(params, 'preconditioner_power')
        precond_power = params.preconditioner_power;  % 0.25 for 1/4, 0.5 for 1/2
    else
        precond_power = 0.25;  % Default: 1/4 power
    end
    
    % Adaptive stepsize settings
    if isfield(params, 'use_adaptive_stepsize')
        use_adaptive_stepsize = params.use_adaptive_stepsize;
    else
        use_adaptive_stepsize = false;  % Default: fixed stepsize
    end
    
    % Stepsize shrinkage settings (alternative to line search)
    if isfield(params, 'use_stepsize_shrinkage')
        use_stepsize_shrinkage = params.use_stepsize_shrinkage;
    else
        use_stepsize_shrinkage = false;  % Default: no shrinkage
    end
    
    if isfield(params, 'shrinkage_factor')
        shrinkage_factor = params.shrinkage_factor;
    else
        shrinkage_factor = 0.99;  % Default: 0.99 contraction per iteration
    end
    
    % Accumulated preconditioner settings (momentum-like for preconditioner)
    if isfield(params, 'use_accumulated_preconditioner')
        use_accumulated_preconditioner = params.use_accumulated_preconditioner;
    else
        use_accumulated_preconditioner = false;  % Default: no accumulation
    end
    
    if isfield(params, 'accumulation_factor')
        accumulation_factor = params.accumulation_factor;
    else
        accumulation_factor = 0.9;  % Default: 0.9 (typical momentum parameter)
    end
    
    if isfield(params, 'line_search_max_iter')
        ls_max_iter = params.line_search_max_iter;
    else
        ls_max_iter = 20;  % Default: max line search iterations
    end
    
    if isfield(params, 'line_search_beta')
        ls_beta = params.line_search_beta;
    else
        ls_beta = 0.5;  % Default: backtracking factor
    end
    
    if isfield(params, 'line_search_c')
        ls_c = params.line_search_c;
    else
        ls_c = 1e-4;  % Default: Armijo constant
    end
    
    % Ground truth
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
        has_ground_truth = true;
    else
        has_ground_truth = false;
    end
    
    % Verbosity
    if isfield(params, 'verbose')
        verbose = params.verbose;
    else
        verbose = 0;
    end
    
    % Spectral initialization
    % if isfield(params, 'use_spectral_init')
    %     use_spectral_init = params.use_spectral_init;
    % else
    %     use_spectral_init = true;
    % end
    
    % Return factorized form
    if isfield(params, 'return_factorized')
        return_factorized = params.return_factorized;
    else
        return_factorized = false;
    end
    
    % Number of measurements
    if nargin < 8 || isempty(m)
        m = length(y);
    end
    
    % Initialize error tracking
    Error_Stand = zeros(T, 1);
    Error_function = zeros(T, 1);
    
    % Track stepsizes if adaptive or shrinking
    if use_adaptive_stepsize || use_stepsize_shrinkage
        stepsize_history = zeros(T, 1);
        if use_adaptive_stepsize
            alpha_prev = alpha;  % Initialize previous stepsize for line search
        end
    end
    
    % Initialize accumulated preconditioner matrices
    if use_accumulated_preconditioner && use_preconditioner
        L_accumulated = [];  % Will be initialized on first iteration
        R_accumulated = [];
    end
    
    % Initialize X_0 = H_r(A^* y) in factorized form: X_0 = U_0 Σ_0 V_0^T
    if isempty(Xl) 
        disp('Initial matrix Xl not provided.');
    else
        % If Xl provided, compute its SVD
        [U_t, Sigma_t, V_t] = rank_r_svd(Xl, r);
    end
    
    % Compute initial errors
    X_current = U_t * Sigma_t * V_t';
    z = operator.A(X_current);
    amplitude_residual = y - abs(z);
    Error_function(1) = (1/(2*m)) * norm(amplitude_residual)^2;
    
    if has_ground_truth
        [Error_Stand(1), ~] = rectify_sign_ambiguity(X_current, Xstar);
    end
    
    if verbose >= 1
        fprintf('RGD-Amplitude: Initial loss = %.6e', Error_function(1));
        if has_ground_truth
            fprintf(', error = %.6e', Error_Stand(1));
        end
        fprintf('\n');
        if use_preconditioner
            fprintf('RGD-Amplitude: Using diagonal preconditioner with ε = %.2e\n', epsilon_reg);
        else
            fprintf('RGD-Amplitude: Standard RGD (no preconditioner)\n');
        end
        if use_adaptive_stepsize
            fprintf('RGD-Amplitude: Using adaptive stepsize (line search)\n');
        elseif use_stepsize_shrinkage
            fprintf('RGD-Amplitude: Using stepsize shrinkage (factor = %.4f)\n', shrinkage_factor);
        else
            fprintf('RGD-Amplitude: Using fixed stepsize α = %.4f\n', alpha);
        end
        if use_accumulated_preconditioner && use_preconditioner
            fprintf('RGD-Amplitude: Using accumulated preconditioner (momentum = %.2f)\n', accumulation_factor);
        end
        fprintf('RGD-Amplitude: Memory: storing %d×%d factorization instead of %d×%d matrix\n', ...
                d1+d2, r, d1, d2);
    end
    
    % RGD iteration loop - all operations on factorized form
    for t = 1:T-1
        % Current X_t = U_t Σ_t V_t^T (never form full matrix unless needed for operators)
        X_current = U_t * Sigma_t * V_t';
        
        % Step 1: Compute gradient of amplitude-based loss
        % G_t = A^*(A(X_t) - y) adapted for amplitude measurements
        % For amplitude: ∇ℓ(X) = -(1/m) * sum_i [(y_i - |z_i|) * sign(z_i)] * A_i^*
        
        z = operator.A(X_current) / sqrt(m);  % Forward measurement
        
        % Compute amplitude residual: y_i - |z_i|
        abs_z = abs(z);
        amplitude_residual = y - abs_z;
        
        % Gradient coefficient: (y_i - |z_i|) * sign(z_i)
        grad_coeff = amplitude_residual .* sign(z);
        
        % Apply adjoint: G_t = -(1/m) * A^*(grad_coeff)
        G_t = operator.A_star(grad_coeff) / sqrt(m);
        G_t = -G_t;
        
        % Ensure gradient is in matrix form
        if isvector(G_t) && (d1 > 1 || d2 > 1)
            G_t = reshape(G_t, [d1, d2]);
        end
        
        % Step 2: Compute preconditioned gradient direction (before stepsize)
        if use_preconditioner
            % Compute current preconditioners:
            % L_t = ε_t I + diag(G_t G_t^T)
            % R_t = ε_t I + diag(G_t^T G_t)
            L_t_current = epsilon_reg + sum(G_t .* G_t, 2);  % diag(G_t G_t^T)
            R_t_current = epsilon_reg + sum(G_t .* G_t, 1)';  % diag(G_t^T G_t)
            
            % Apply accumulation if enabled (momentum-like for preconditioner)
            if use_accumulated_preconditioner
                if isempty(L_accumulated)
                    % First iteration: initialize with current values
                    L_accumulated = L_t_current;
                    R_accumulated = R_t_current;
                else
                    % Accumulate: L_new = (1-c) * L_old + c * L_current
                    % This smooths the preconditioner over iterations
                    L_accumulated = (1 - accumulation_factor) * L_accumulated + accumulation_factor * L_t_current;
                    R_accumulated = (1 - accumulation_factor) * R_accumulated + accumulation_factor * R_t_current;
                end
                % Use accumulated values
                L_t_diag = L_accumulated;
                R_t_diag = R_accumulated;
            else
                % Use current values directly
                L_t_diag = L_t_current;
                R_t_diag = R_t_current;
            end
            
            % Compute L_t^{-1/p} and R_t^{-1/p} for gradient preconditioning
            % where p is the preconditioner power (0.25 for 1/4, 0.5 for 1/2)
            L_t_inv_power = L_t_diag .^ (-precond_power);
            R_t_inv_power = R_t_diag .^ (-precond_power);
            
            % Compute search direction (without stepsize): D_t = L_t^{-1/p} G_t R_t^{-1/p}
            D_t = (L_t_inv_power * ones(1, d2)) .* G_t .* (ones(d1, 1) * R_t_inv_power');
        else
            % No preconditioner: D_t = G_t
            D_t = G_t;
            L_t_diag = [];
            R_t_diag = [];
        end
        
        % Step 2b: Determine stepsize (adaptive, shrinking, or fixed)
        if use_adaptive_stepsize
            % Adaptive stepsize via backtracking line search
            % Find α that satisfies Armijo condition:
            % ℓ(X_t - α*D_t) ≤ ℓ(X_t) - c*α*<∇ℓ(X_t), D_t>
            
            % Smart initialization: start from previous successful stepsize
            % Optionally try a slightly larger stepsize (1.2x growth factor)
            alpha_t = min(1.05 * alpha_prev, 5 * alpha);  % Grow but cap at 5x initial
            current_loss = Error_function(t);
            
            % Directional derivative: <G_t, D_t> (Frobenius inner product)
            direc_deriv = sum(G_t(:) .* D_t(:));
            
            % Backtracking line search
            for ls_iter = 1:ls_max_iter
                % Try step with current alpha_t
                Z_trial = -alpha_t * D_t;
                
                % Compute trial point
                [U_trial, Sigma_trial, V_trial] = efficient_rank_r_update(...
                    U_t, Sigma_t, V_t, Z_trial, r, use_preconditioner, L_t_diag, R_t_diag, precond_power);
                
                X_trial = U_trial * Sigma_trial * V_trial';
                z_trial = operator.A(X_trial) / sqrt(m);  % Normalize measurements
                residual_trial = y - abs(z_trial);
                loss_trial = (1/(2)) * norm(residual_trial)^2;
                
                % Check Armijo condition
                if loss_trial <= current_loss - ls_c * alpha_t * direc_deriv
                    break;  % Accept this stepsize
                end
                
                % Reduce stepsize
                alpha_t = ls_beta * alpha_t;
            end
            
            % Store stepsize for tracking
            stepsize_history(t) = alpha_t;
            alpha_prev = alpha_t;  % Remember for next iteration
            
            if verbose >= 3
                fprintf('  Line search: α = %.4e (after %d iterations, started from %.4e)\n', ...
                        alpha_t, ls_iter, min(1.2 * alpha_prev, 5 * alpha));
            end
        elseif use_stepsize_shrinkage
            % Shrinking stepsize: α_t = α_0 * γ^t
            alpha_t = alpha * (shrinkage_factor ^ (t-1));
            
            % Store stepsize for tracking
            stepsize_history(t) = alpha_t;
            
            if verbose >= 3
                fprintf('  Shrinking stepsize: α = %.4e (α_0 * %.4f^%d)\n', ...
                        alpha_t, shrinkage_factor, t-1);
            end
        else
            % Fixed stepsize
            alpha_t = alpha;
        end
        
        % Compute Z_t = -α_t * D_t with chosen stepsize
        Z_t = -alpha_t * D_t;
        
        % Step 3: Compute H_r(W_t) = H_r(X_t + P̃_T(Z_t)) directly
        % Without forming Ũ_t, Ṽ_t, D_t, or W_t
        % Use efficient algorithm: W_t = [U_t Q_2] M_t [V_t Q_1]^T
        [U_t, Sigma_t, V_t] = efficient_rank_r_update(U_t, Sigma_t, V_t, Z_t, r, ...
                                                       use_preconditioner, L_t_diag, R_t_diag, precond_power);
        
        % Step 4: Compute errors for iteration t+1
        X_new = U_t * Sigma_t * V_t';
        z_new = operator.A(X_new)/sqrt(m);
        amplitude_residual_new = y - abs(z_new);
        Error_function(t+1) = (1/(2)) * norm(amplitude_residual_new)^2;

        if has_ground_truth
            [Error_Stand(t+1), ~] = rectify_sign_ambiguity(X_new, Xstar);
        end
        
        % Optional: Print progress
        if verbose >= 2 || (verbose >= 1 && mod(t, 50) == 0)
            fprintf('  Iter %4d: Loss = %.6e', t, Error_function(t+1));
            if has_ground_truth
                fprintf(', Rel. Error = %.6e', Error_Stand(t+1));
            end
            fprintf(', rank = %d\n', size(U_t, 2));
        end
    end
    
    if verbose >= 1
        fprintf('RGD-Amplitude: Final loss = %.6e', Error_function(end));
        if has_ground_truth
            fprintf(', Final error = %.6e', Error_Stand(end));
        end
        fprintf(', Final rank = %d\n', size(U_t, 2));
    end
    
    % Pack output struct
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
    output.use_preconditioner = use_preconditioner;
    output.epsilon = epsilon_reg;
    output.rank = size(U_t, 2);
    output.U = U_t;
    output.Sigma = Sigma_t;
    output.V = V_t;
    
    % Add stepsize history if adaptive or shrinking
    if use_adaptive_stepsize || use_stepsize_shrinkage
        output.stepsize_history = stepsize_history;
    end
    
    % Return full matrix or factorized form
    if return_factorized
        % Return struct with SVD factors
        Xl = struct('U', U_t, 'Sigma', Sigma_t, 'V', V_t);
    else
        % Return full matrix
        Xl = U_t * Sigma_t * V_t';
    end
end

%% Helper Functions

function [U_r, Sigma_r, V_r] = rank_r_svd(X, r)
    % Compute rank-r SVD truncation: X_r = U_r Σ_r V_r^T
    % Returns factors instead of full matrix
    
    % Check if symmetric for efficiency
    is_symmetric = (size(X, 1) == size(X, 2)) && norm(X - X', 'fro') < 1e-10 * norm(X, 'fro');
    
    if is_symmetric
        % Use eigendecomposition for symmetric matrices
        [U, S] = eig(X);
        [~, idx] = sort(abs(diag(S)), 'descend');
        U = U(:, idx);
        S = S(idx, idx);
        
        % Truncate to rank r
        r_actual = min(r, size(S, 1));
        U_r = U(:, 1:r_actual);
        Sigma_r = S(1:r_actual, 1:r_actual);
        V_r = U_r;  % Symmetric case: V = U
    else
        % Use SVD for general matrices
        [U, S, V] = svd(X, 'econ');
        
        % Truncate to rank r
        r_actual = min(r, size(S, 1));
        U_r = U(:, 1:r_actual);
        Sigma_r = S(1:r_actual, 1:r_actual);
        V_r = V(:, 1:r_actual);
    end
end

function [U_new, Sigma_new, V_new] = efficient_rank_r_update(U_t, Sigma_t, V_t, Z_t, r, ...
                                                              use_precond, L_diag, R_diag, precond_power)
    % Efficient computation of H_r(W_t) where W_t = X_t + P̃_T(Z_t)
    % Directly computes the result without forming Ũ_t, Ṽ_t, D_t, or W_t
    %
    % Following the derivation:
    % W_t = U_t Σ_t V_t^T + P̃_T(Z_t)
    %     = U_t K_0 V_t^T + U_t Y_1^T + Y_2 V_t^T
    %     = [U_t Q_2] M_t [V_t Q_1]^T
    %
    % where M_t is a small 2r×2r matrix, and we compute H_r(W_t) via SVD(M_t)
    % precond_power: power for preconditioner (0.25 for 1/4, 0.5 for 1/2, default: 0.25)
    
    % Handle both preconditioned and non-preconditioned cases uniformly
    % Non-preconditioned case: treat as L_t = I, R_t = I (identity matrices)
    
    [n1, ~] = size(U_t);
    [n2, ~] = size(V_t);
    
    if ~use_precond
        % No preconditioner: L_t = I, R_t = I
        % So L_t^{1/p} = I and R_t^{1/p} = I (diagonal matrices of ones)
        L_power = ones(n1, 1);
        R_power = ones(n2, 1);
    else
        % With preconditioner: compute L_t^{1/p} and R_t^{1/p}
        L_power = L_diag .^ precond_power;
        R_power = R_diag .^ precond_power;
    end
    
    % Compute M1 = U_t^T L_t^{1/p} U_t and M2 = V_t^T R_t^{1/p} V_t (both r×r)
    M1 = U_t' * diag(L_power) * U_t;
    M2 = V_t' * diag(R_power) * V_t;
    
    % Compute M1^{-1} and M2^{-1} via direct inverse
    % M1_inv = inv(M1);
    % M2_inv = inv(M2);
    
    % Precompute needed terms
    U_T_L_power = U_t' * diag(L_power);  % r×n1
    Z_R_power = Z_t * diag(R_power);      % n1×n2
    
    % Compute K_0 (r×r matrix):
    % K_0 = Σ_t + M1^{-1} U_t^T L_t^{1/p} Z_t V_t 
    %           + (U_t^T - M1^{-1} U_t^T L_t^{1/p}) Z_t R_t^{1/p} V_t M2^{-1}

    term1 = (M1 \ (U_T_L_power * Z_t * V_t));
    term2 = ((U_t' - M1 \ U_T_L_power) * Z_R_power * V_t) / M2;
    K_0 = Sigma_t + term1 + term2;
    
    % Compute Y_1^T (r×n2 matrix):
    % Y_1^T = M1^{-1} U_t^T L_t^{1/p} Z_t (I - V_t V_t^T)
    Y_1_T = (M1 \ (U_T_L_power * Z_t * (eye(n2) - V_t * V_t')));

    % term1 = M1_inv * U_T_L_quarter * Z_t * V_t;
    % term2 = (U_t' - M1_inv * U_T_L_quarter) * Z_R_quarter * V_t * M2_inv;
    % K_0 = Sigma_t + term1 + term2;
    
    % % Compute Y_1^T (r×n2 matrix):
    % % Y_1^T = M1^{-1} U_t^T L_t^{1/4} Z_t (I - V_t V_t^T)
    % Y_1_T = M1_inv * U_T_L_quarter * Z_t * (eye(n2) - V_t * V_t');
    
    % Compute Y_2 (n1×r matrix):
    % Y_2 = (I - U_t U_t^T) Z_t R_t^{1/p} V_t M2^{-1}
    %Y_2 = (eye(n1) - U_t * U_t') * Z_R_quarter * V_t * M2_inv;
    Y_2 = (eye(n1) - U_t * U_t') * Z_R_power * V_t / M2;
    
    % QR factorizations to get orthonormal bases
    % Y_1 = Q_1 K_1^T  (Y_1 is n2×r, so Y_1^T is r×n2)
    [Q_1, K_1] = qr(Y_1_T', 0);  % Q_1: n2×r', K_1: r'×r
    
    % Y_2 = Q_2 K_2    (Y_2 is n1×r)
    [Q_2, K_2] = qr(Y_2, 0);     % Q_2: n1×r', K_2: r'×r
    
    % Form the small matrix M_t:
    % M_t = [K_0    K_1^T]
    %       [K_2    0    ]
    % M_t is (r + r')×(r + r') where r' = rank of Y_1, Y_2 (at most r)
    M_t = [K_0, K_1'; 
           K_2, zeros(size(K_2, 2), size(K_1, 2))];
    
    % Compute SVD of M_t (small matrix, O(r^3) cost)
    [U_M, S_M, V_M] = svd(M_t, 'econ');
    
    % Truncate to rank r
    r_actual = min(r, size(S_M, 1));
    U_M_r = U_M(:, 1:r_actual);
    S_M_r = S_M(1:r_actual, 1:r_actual);
    V_M_r = V_M(:, 1:r_actual);
    
    % Reconstruct U_new and V_new from the factorization:
    % W_t = [U_t Q_2] M_t [V_t Q_1]^T
    % So SVD(W_t) = [U_t Q_2] U_M_r S_M_r V_M_r^T [V_t Q_1]^T
    U_new = [U_t, Q_2] * U_M_r;
    V_new = [V_t, Q_1] * V_M_r;
    Sigma_new = S_M_r;
end
