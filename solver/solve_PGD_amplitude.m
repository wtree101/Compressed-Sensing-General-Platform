function [output, Xl] = solve_PGD_amplitude(Xl, ~, y, operator, d1, d2, ~, m, params)
    % solve_PGD_amplitude - Projected Gradient Descent for Amplitude-Based Loss
    % 
    % This solver minimizes the amplitude-based loss:
    %   ℓ(X) = (1/2m) * sum_i (y_i - |<A_i, X>|)^2
    % 
    % Iteration rule:
    %   X_tilde^(t+1) = X^(t) - η * ∇ℓ(X^(t))
    %   X^(t+1) = Projection_{rank(X)≤r}(X_tilde^(t+1))
    %
    % Inputs:
    %   Xl - Initial matrix (d1 x d2)
    %   ~ - Unused (for compatibility)
    %   y - Measurement vector (magnitudes)
    %   operator - Struct containing A (forward) and A_star (adjoint) operators
    %   d1, d2 - Matrix dimensions
    %   ~ - Unused rank parameter (now in params)
    %   m - Number of measurements (or can be computed from y)
    %   params - Parameter structure containing:
    %       - T: number of iterations
    %       - mu (or eta): step size
    %       - projection: projection function handle
    %       - use_preconditioner: enable/disable preconditioning (default: false)
    %       - epsilon: regularization for preconditioner (default: 1e-8)
    %       - Xstar: ground truth (for error tracking)
    %
    % Outputs:
    %   output - Struct containing auxiliary information:
    %            .Error_Stand    - Standard error tracking
    %            .Error_function - Function error tracking
    %   Xl - Final solution
    
    % Extract parameters
    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
    
    if isfield(params, 'mu')
        eta = params.mu;  % Step size
    elseif isfield(params, 'eta')
        eta = params.eta;
    else
        eta = 0.1;  % Default step size
    end
    
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
        has_ground_truth = true;
    else
        has_ground_truth = false;
        warning('Ground truth Xstar not provided. Error tracking will be disabled.');
    end
    
    if ~isfield(params, 'projection') || isempty(params.projection)
        error('Projection function must be provided in params.projection');
    end
    
    % Preconditioner settings
    if isfield(params, 'use_preconditioner')
        use_preconditioner = params.use_preconditioner;
    else
        use_preconditioner = false;  % Default: no preconditioner for PGD
    end
    
    if isfield(params, 'epsilon')
        epsilon_reg = params.epsilon;
    else
        epsilon_reg = 1e-8;  % Default regularization
    end
    
    % Adaptive stepsize settings
    if isfield(params, 'use_adaptive_stepsize')
        use_adaptive_stepsize = params.use_adaptive_stepsize;
    else
        use_adaptive_stepsize = false;  % Default: fixed stepsize
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
    
    % Number of measurements
    if nargin < 8 || isempty(m)
        m = length(y);
    end
    
    % Initialize error tracking
    Error_Stand = zeros(T, 1);
    Error_function = zeros(T, 1);
    
    % Track stepsizes if adaptive
    if use_adaptive_stepsize
        stepsize_history = zeros(T, 1);
        eta_prev = eta;  % Initialize previous stepsize
    end
    
    % Compute initial errors
    z = operator.A(Xl)/sqrt(m);
    amplitude_residual = y - abs(z);
    Error_function(1) = (1/(2)) * norm(amplitude_residual)^2;
    
    if has_ground_truth
        [Error_Stand(1), ~] = rectify_sign_ambiguity(Xl, Xstar);
    end
    
    %fprintf('PGD-Amplitude: Initial loss = %.6e\n', Error_function(1));
    % if use_preconditioner
    %     fprintf('PGD-Amplitude: Using diagonal preconditioner with ε = %.2e\n', epsilon_reg);
    % else
    %     fprintf('PGD-Amplitude: Standard PGD (no preconditioner)\n');
    % end
    
    % PGD iteration loop
    for t = 1:T-1
        % Step 1: Compute gradient of amplitude-based loss
        % ∇ℓ(X) = -(1/m) * sum_i [(y_i - |z_i|) / |z_i|] * sign(z_i) * A_i^*
        % where z_i = <A_i, X>
        
        z = operator.A(Xl)/sqrt(m);  % Forward measurement: z = A(X)
        
        % Compute amplitude residual: y_i - |z_i|
        abs_z = abs(z);
        amplitude_residual = y - abs_z;
        
        % Avoid division by zero: add small epsilon where |z_i| is small
        epsilon = 1e-12;
        safe_abs_z = abs_z + epsilon;
        
        % Gradient coefficient: (y_i - |z_i|) / |z_i| * conj(z_i) / |z_i|
        %                      = (y_i - |z_i|) * conj(z_i) / |z_i|^2
        grad_coeff = amplitude_residual .* sign(z);
        
        % Apply adjoint: ∇ℓ(X) = -(1/m) * A^*(grad_coeff)
        gradient = operator.A_star(grad_coeff) / sqrt(m);
        gradient = - gradient;
        
        % Ensure gradient is in matrix form
        if isvector(gradient) && (d1 > 1 || d2 > 1)
            gradient = reshape(gradient, [d1, d2]);
        end
        
        % Step 2: Apply preconditioner (if enabled)
        if use_preconditioner
            % Update diagonal preconditioners:
            % L_t = ε_t I + diag(G_t G_t^T)
            % R_t = ε_t I + diag(G_t^T G_t)
            L_t_diag = epsilon_reg + sum(gradient .* gradient, 2);  % diag(G_t G_t^T)
            R_t_diag = epsilon_reg + sum(gradient .* gradient, 1)';  % diag(G_t^T G_t)
            
            % Compute L_t^{-1/4} and R_t^{-1/4} for gradient preconditioning
            L_t_inv_quarter = L_t_diag .^ (-0.25);
            R_t_inv_quarter = R_t_diag .^ (-0.25);
            
            % Apply preconditioning: D_t = L_t^{-1/4} G_t R_t^{-1/4}
            preconditioned_gradient = (L_t_inv_quarter * ones(1, d2)) .* gradient .* (ones(d1, 1) * R_t_inv_quarter');
        else
            % No preconditioner: use gradient directly
            preconditioned_gradient = gradient;
        end
        
        % Step 2b: Determine stepsize (adaptive or fixed)
        if use_adaptive_stepsize
            % Adaptive stepsize via backtracking line search
            % Find η that satisfies Armijo condition:
            % ℓ(X_t - η*D_t) ≤ ℓ(X_t) - c*η*<∇ℓ(X_t), D_t>
            
            % Smart initialization: start from previous successful stepsize
            % Optionally try a slightly larger stepsize (1.05x growth factor)
            eta_t = min(1.05 * eta_prev, 5 * eta);  % Grow but cap at 5x initial
            current_loss = Error_function(t);
            
            % Directional derivative: <gradient, preconditioned_gradient> (Frobenius inner product)
            direc_deriv = sum(gradient(:) .* preconditioned_gradient(:));
            
            % Backtracking line search
            for ls_iter = 1:ls_max_iter
                % Try step with current eta_t
                Xl_trial = Xl - eta_t * preconditioned_gradient;
                
                % Apply projection
                Xl_trial = params.projection(Xl_trial);
                
                % Compute trial loss
                z_trial = operator.A(Xl_trial) / sqrt(m);
                residual_trial = y - abs(z_trial);
                loss_trial = (1/(2)) * norm(residual_trial)^2;
                
                % Check Armijo condition
                if loss_trial <= current_loss - ls_c * eta_t * direc_deriv
                    break;  % Accept this stepsize
                end
                
                % Reduce stepsize
                eta_t = ls_beta * eta_t;
            end
            
            % Store stepsize for tracking
            stepsize_history(t) = eta_t;
            eta_prev = eta_t;  % Remember for next iteration
        else
            % Fixed stepsize
            eta_t = eta;
        end
        
        % Step 3: Gradient descent step with chosen stepsize
        Xl_temp = Xl - eta_t * preconditioned_gradient;
        
        % Step 4: Projection onto constraint set (e.g., rank-r matrices)
        Xl = params.projection(Xl_temp);
        
        % Step 5: Compute errors for iteration t+1
        z_new = operator.A(Xl)/sqrt(m);
        amplitude_residual_new = y - abs(z_new);
        Error_function(t+1) = (1/(2)) * norm(amplitude_residual_new)^2;
        
        if has_ground_truth
            [Error_Stand(t+1), ~] = rectify_sign_ambiguity(Xl, Xstar);
        end
        
        % % Optional: Print progress
        % if mod(t, 50) == 0 || t == 1
        %     if has_ground_truth
        %         fprintf('  Iter %4d: Loss = %.6e, Rel. Error = %.6e\n', ...
        %             t, Error_function(t+1), Error_Stand(t+1));
        %     else
        %         fprintf('  Iter %4d: Loss = %.6e\n', t, Error_function(t+1));
        %     end
        % end
    end
    
    % fprintf('PGD-Amplitude: Final loss = %.6e\n', Error_function(end));
    % if has_ground_truth
    %     fprintf('PGD-Amplitude: Final relative error = %.6e\n', Error_Stand(end));
    % end
    
    % Pack output struct
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
    output.use_preconditioner = use_preconditioner;
    output.epsilon = epsilon_reg;
    
    % Add stepsize history if adaptive
    if use_adaptive_stepsize
        output.stepsize_history = stepsize_history;
    end
end
