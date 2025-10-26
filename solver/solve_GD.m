function [output, Xl] = solve_GD(Xl, Ul, y, operator, d1, d2, r, m, params)
    % GD Solver - Standard Gradient Descent
    % Inputs:
    %   Xl - Initial matrix (or general object)
    %   Ul - Initial factorization (unused in this solver, for compatibility)
    %   y - Measurement vector
    %   operator - Struct containing A (forward) and A_star (adjoint) operators
    %   d1, d2 - Matrix dimensions
    %   r - Rank
    %   m - Number of measurements
    %   params - Parameter structure containing mu, lambda, T, etc.
    % Outputs:
    %   output - Struct containing auxiliary information:
    %            .Error_Stand    - Standard error tracking
    %            .Error_function - Function error tracking
    %   Xl - Final recovered matrix
    
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
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
    else
        disp('Error: Xstar (ground truth) must be provided.');
    end
    
    % Initialize Ul if not provided or empty
    if isempty(Ul)
        [U, ~, ~] = svd(Xl, 'econ');
        Ul = U(:, 1:min(r, size(U, 2)));
    end
    
    m = length(y);
   
    
    % Error Tracking
    Error_Stand = zeros(T,1);
    Error_Stand(1) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    Error_function = zeros(T,1);
    Error_function(1) = norm(y - operator.A(Xl)/sqrt(length(y)),'fro')/norm(y);
    
    % GD optimization loop
    for l = 2:T
    % compute Gl
    
        yl = operator.A(Xl) / sqrt(m);
        
        s = y - operator.A(Xl) / sqrt(m);
    
        % Apply adjoint operator and reshape if needed
        grad_term = operator.A_star(s);
        %grad_term = A' * s;
        if isvector(grad_term)
            grad_term = reshape(grad_term, [d1, d2]);
        end

        Ul = Ul + mu* ( (1*(1/sqrt(m)) * grad_term)*Ul - lambda*Ul); 
        %Xl_new = Ul * Ul'; % Update Xl
        % Track Errors
        
        % Swap 
        Xl = Ul*Ul';
        Error_function(l) = norm(yl - y, 'fro')^2 + lambda * (norm(Xl, 'fro')^2 -  norm(Xstar, 'fro')^2) / 2;
        Error_Stand(l) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    end
    
    % Pack output struct
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
end
