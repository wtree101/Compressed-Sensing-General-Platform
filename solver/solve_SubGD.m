function [Error_Stand, Error_function, Xl] = solve_SubGD(Xl, Ul, y, operator, d1, d2, r, m, params)
    % SubGD Solver - Subgradient Descent
    % Inputs:
    %   Xl, Ul - Initial matrices (or general objects)
    %   y - Measurement vector
    %   operator - Struct containing A (forward) and A_star (adjoint) operators
    %   d1, d2 - Matrix dimensions
    %   r - Rank
    %   m - Number of measurements
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
        mu=0.1;
    end
    if isfield(params, 'lambda')
        lambda = params.lambda;
    else
        lambda=0;
    end
     if isfield(params, 'Xstar')
        Xstar = params.Xstar;
    else
        disp('Error: Xstar (ground truth) must be provided.');
    end
    
    % Error Tracking
    Error_Stand = zeros(T,1);
    Error_Stand(1) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    Error_function = zeros(T,1);
    Error_function(1) = norm(y - operator.A(Xl)/sqrt(length(y)),'fro')/norm(y);
    
    % SubGD optimization loop
    for l = 2:T
        % Compute current prediction using operator
        yl = operator.A(Xl) / sqrt(length(y));
        
        % Compute residual
        s = y - yl;
        
        % Compute gradient using adjoint operator
        Gl = (1/sqrt(length(y))) * operator.A_star(s);
        
        % Ensure Gl is properly shaped as matrix
        if isvector(Gl)
            Gl = reshape(Gl, [d1, d2]);
        end
        
        % Update Ul using SubGD rule (project gradient onto subspace)
        Ul = Ul + mu * (Gl * Ul - lambda * Ul);
        
        % Update Xl
        Xl = Ul * Ul';
        
        % Track Errors
        Error_function(l) = norm(yl - y, 'fro')^2 + lambda * (norm(Xl, 'fro')^2 - norm(Xstar, 'fro')^2) / 2;
        Error_Stand(l) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    end
end
