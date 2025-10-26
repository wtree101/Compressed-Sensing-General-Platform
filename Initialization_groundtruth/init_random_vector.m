function xl = init_random_vector(y, operator, d1, sparsity, m, scale)
    % Initialization for sparse vector recovery (compressed sensing)
    % Inputs:
    %   y - Measurement vector
    %   operator - Sensing operator structure
    %   d1 - Vector dimension
    %   sparsity - Expected sparsity level
    %   m - Number of measurements
    %   scale - (optional) Initialization scale
    % Output:
    %   xl - Initialized vector
    
    if nargin < 6
        scale = 1e-3;
    end
    
    % Method 1: Zero initialization (most common for sparse recovery)
    if m < d1
        xl = zeros(d1, 1);
        return;
    end
    
    % Method 2: Least squares initialization (if m >= d1)
    try
        % Use adjoint for initial estimate
        xl_ls = operator.A_star(y);
        
        % Scale appropriately
        if norm(xl_ls) > 0
            xl = xl_ls / norm(xl_ls) * scale * sqrt(sparsity);
        else
            xl = zeros(d1, 1);
        end
        
        % Optional: Apply initial sparsity by keeping largest components
        if sparsity < d1
            [~, idx] = maxk(abs(xl), sparsity);
            xl_sparse = zeros(d1, 1);
            xl_sparse(idx) = xl(idx);
            xl = xl_sparse;
        end
        
    catch
        % Fallback to zero initialization
        xl = zeros(d1, 1);
    end
end
