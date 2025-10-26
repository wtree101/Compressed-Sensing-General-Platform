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
    
    % Number of measurements
    if nargin < 8 || isempty(m)
        m = length(y);
    end
    
    % Initialize error tracking
    Error_Stand = zeros(T, 1);
    Error_function = zeros(T, 1);
    
    % Compute initial errors
    z = operator.A(Xl);
    amplitude_residual = y - abs(z);
    Error_function(1) = (1/(2*m)) * norm(amplitude_residual)^2;
    
    if has_ground_truth
        [Error_Stand(1), ~] = rectify_sign_ambiguity(Xl, Xstar);
    end
    
    %fprintf('PGD-Amplitude: Initial loss = %.6e\n', Error_function(1));
    
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
        
        % Step 2: Gradient descent step
        Xl_temp = Xl - eta * gradient;
        
        % Step 3: Projection onto constraint set (e.g., rank-r matrices)
        Xl = params.projection(Xl_temp);
        
        % Step 4: Compute errors for iteration t+1
        z_new = operator.A(Xl);
        amplitude_residual_new = y - abs(z_new);
        Error_function(t+1) = (1/(2*m)) * norm(amplitude_residual_new)^2;
        
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
end
