function [Error_Stand, Error_function, xl] = solve_GD_vec(xl, ~, y, operator, d1, ~, sparsity, m, params)
    % Proximal Gradient Descent Solver for Sparse Vector Recovery (ISTA)
    % Inputs:
    %   xl - Initial vector
    %   ~ - Unused (for compatibility with matrix interface)
    %   y - Measurement vector
    %   operator - Struct containing A (forward) and A_star (adjoint) operators
    %   d1 - Vector dimension
    %   ~ - Unused (d2 not needed for vectors)
    %   sparsity - Sparsity level
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
        mu = 0.1; % Step size for proximal gradient
    end
    if isfield(params, 'lambda')
        lambda = params.lambda;
    else
        lambda = 0.01; % L1 regularization parameter
    end
    if isfield(params, 'xstar')
        xstar = params.xstar;
    else
        disp('Warning: xstar (ground truth) not provided for error tracking.');
        xstar = zeros(d1, 1); % Dummy ground truth
    end
    
    % Error Tracking
    Error_Stand = zeros(T, 1);
    if norm(xstar) > 0
        Error_Stand(1) = norm(xl - xstar) / norm(xstar);
    else
        Error_Stand(1) = norm(xl);
    end
    Error_function = zeros(T, 1);
    Error_function(1) = norm(y - operator.A(xl)/sqrt(m)) / norm(y);
    
    % Proximal Gradient Descent optimization loop (ISTA)
    for l = 2:T
        % Compute current prediction
        yl = operator.A(xl) / sqrt(m);
        
        % Compute residual
        s = y - yl;
        
        % Compute gradient of smooth part (data fidelity)
        grad = operator.A_star(s) / sqrt(m);
        
        % Gradient step
        z = xl + mu * grad;
        
        % Proximal operator for L1 norm (soft thresholding)
        threshold = lambda * mu;
        xl = soft_threshold(z, threshold);
        
        % Track Errors
        if norm(xstar) > 0
            Error_Stand(l) = norm(xl - xstar) / norm(xstar);
        else
            Error_Stand(l) = norm(xl);
        end
        
        yl = operator.A(xl)/sqrt(m);
        Error_function(l) = norm(y - yl) / norm(y);
        
        % Early stopping if error is too large
        if Error_Stand(l) > 1e3
            break;
        end
    end
end

function x_thresh = soft_threshold(x, threshold)
    % Soft thresholding operator (proximal operator for L1 norm)
    % Input:
    %   x - Input vector
    %   threshold - Thresholding parameter
    % Output:
    %   x_thresh - Soft-thresholded vector
    
    x_thresh = sign(x) .* max(abs(x) - threshold, 0);
end
