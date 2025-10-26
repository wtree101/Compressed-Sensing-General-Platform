function xl = init_zero_vector(y, operator, d1, sparsity, m, scale)
    % Zero initialization for sparse vector recovery
    % This is the most common initialization for compressed sensing
    % Inputs:
    %   y, operator, d1, sparsity, m, scale - (unused but kept for consistency)
    % Output:
    %   xl - Zero vector
    
    xl = zeros(d1, 1);
end
