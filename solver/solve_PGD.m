function [output, Xl] = solve_PGD(Xl, ~, y, operator, d1, d2, ~, m, params)
    % PGD Solver - Projected Gradient Descent
    % Inputs:
    %   Xl - Initial matrix (or general object)
    %   ~ - Unused (for compatibility)
    %   y - Measurement vector
    %   operator - Struct containing A (forward) and A_star (adjoint) operators
    %   d1, d2 - Matrix dimensions
    %   ~ - Rank constraint (unused, read from params.r)
    %   m - Number of measurements
    %   params - Parameter structure containing mu, lambda, T, etc.
    % Outputs:
    %   output - Struct containing auxiliary information:
    %            .Error_Stand    - Standard error tracking
    %            .Error_function - Function error tracking
    %   Xl - Final projected solution
    
    % Extract parameters from params structure
    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
    if isfield(params, 'mu')
        mu = params.mu;
    else
        mu = 0.01; % Smaller step size for PGD
    end
    if isfield(params, 'lambda')
        lambda = params.lambda;
    else
        lambda = 0;
    end
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
    else
        disp('Error: Xstar (ground truth) must be provided.');
    end
    
    m = length(y);
    
    % Error Tracking
    Error_Stand = zeros(T,1);
    Error_Stand(1) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    Error_function = zeros(T,1);
    
    % Initial function value
    yl = operator.A(Xl) / sqrt(m);
    Error_function(1) = 0.5 * norm(y - yl, 'fro')^2 + 0.5 * lambda * norm(Xl, 'fro')^2;
    
    % PGD optimization loop
    for l = 2:T
        % Step 1: Compute gradient of loss function
        % Loss: f(X) = 0.5 * ||A(X)/sqrt(m) - y||^2 + 0.5 * lambda * ||X||_F^2
        
        % Forward pass
        yl = operator.A(Xl) / sqrt(m);
        residual = yl - y;
        
        % Compute gradient: ∇f(X) = A^*(A(X)/sqrt(m) - y)/sqrt(m) + lambda * X
        grad_data = operator.A_star(residual) / sqrt(m);
        if isvector(grad_data)
            grad_data = reshape(grad_data, [d1, d2]);
        end
        
        gradient = grad_data + lambda * Xl;
        
        % Step 2: Gradient descent step
        Xl_temp = Xl - mu * gradient;
        
        % Step 3: Projection onto rank-r constraint set
        Xl = params.projection(Xl_temp);
        
        % Step 4: Track errors
        yl_new = operator.A(Xl) / sqrt(m);
        Error_function(l) = 0.5 * norm(y - yl_new, 'fro')^2 + 0.5 * lambda * norm(Xl, 'fro')^2;
        Error_Stand(l) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    end
    
    % Pack output struct
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
end

