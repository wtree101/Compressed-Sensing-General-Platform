function cond_ratio = cond_ratio(X,r)
    % Compute the ratio of the smallest singular value to the largest singular value.
    % Inputs:
    %   X - Input matrix
    % Outputs:
    %   cond_ratio - Condition number defined as sigma(r) / sigma(1)
    
    % Compute the singular values of X
    singular_values = svd(X);
    
    % Compute the ratio sigma(r) / sigma(1)
    cond_ratio = singular_values(1) / singular_values(r);
end