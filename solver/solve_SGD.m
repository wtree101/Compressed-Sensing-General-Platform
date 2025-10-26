function [Error_Stand, Error_function, Xl] = solve_SGD(Xl, Ul, y, A, d1, d2, r, m, params)
    % SGD Solver - Stochastic Gradient Descent  
    % Inputs:
    %   Xl, Ul - Initial matrices
    %   y - Measurement vector
    %   A - Sensing matrix
    %   d1, d2 - Matrix dimensions
    %   r - Rank
    %   m - Number of measurements
    %   params - Parameter structure containing mu, lambda, T, etc.[Error_Stand, Error_function] = solve_SGD(Xl, Ul, y, A, Xstar, d1, d2, params)
    % SGD Solver - Stochastic Gradient Descent
    % Inputs:
    %   Xl, Ul - Initial matrices
    %   y - Measurement vector
    %   A - Sensing matrix
    %   Xstar - Ground truth matrix
    %   d1, d2 - Matrix dimensions
    %   params - Parameter structure containing mu, lambda, T, etc.
    % Outputs:
    %   Error_Stand - Standard error tracking
    %   Error_function - Function error tracking
    
    % Extract parameters from params structure
    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
    if isfield(params, 'mu')
        mu = params.mu;
    else
        mu=0.2;
    end
    if isfield(params, 'lambda')
        lambda = params.lambda;
    else
        lambda=0;
    end
    
    
    m = length(y);
    
    % Error Tracking
    Error_Stand = zeros(T,1);
    Error_Stand(1) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    Error_function = zeros(T,1);
    Error_function(1) = norm(y - A*Xl(:)/sqrt(m),'fro')/norm(y);
    
    % SGD optimization loop
    for l = 2:T
        % Randomly sample a measurement
        idx = randi(m);
        
        % Compute prediction for this sample
        yl_i = A(idx,:) * Xl(:) / sqrt(m);
        
        % Compute residual for this sample
        s_i = y(idx) - yl_i;
        
        % Compute stochastic gradient
        Gl = (1/sqrt(m)) * s_i * reshape(A(idx,:), [d1, d2]);
        
        % Update Ul using SGD rule
        Ul = Ul + mu * (Gl * Ul - lambda * Ul);
        
        % Update Xl
        Xl = Ul * Ul';
        
        % Track Errors (compute full error for tracking)
        yl_full = A * Xl(:) / sqrt(m);
        Error_function(l) = norm(yl_full - y, 'fro')^2 + lambda * (norm(Xl, 'fro')^2 - norm(Xstar, 'fro')^2) / 2;
        Error_Stand(l) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    end
end
