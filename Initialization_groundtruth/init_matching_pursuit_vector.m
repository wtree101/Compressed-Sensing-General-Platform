function xl = init_matching_pursuit_vector(y, operator, d1, sparsity, m, scale)
    % Matching Pursuit initialization for sparse vector recovery
    % Selects the most correlated atoms with the measurements
    % Inputs:
    %   y - Measurement vector
    %   operator - Sensing operator structure
    %   d1 - Vector dimension
    %   sparsity - Expected sparsity level
    %   m, scale - Optional parameters
    % Output:
    %   xl - Initialized sparse vector
    
    if nargin < 6
        scale = 1e-2;
    end
    
    try
        % Compute correlations: A^T * y
        correlations = operator.A_star(y);
        
        % Find the sparsity largest correlations
        [~, idx] = maxk(abs(correlations), min(sparsity, d1));
        
        % Initialize sparse vector
        xl = zeros(d1, 1);
        xl(idx) = correlations(idx) * scale;
        
    catch
        % Fallback to zero initialization
        xl = zeros(d1, 1);
    end
end
