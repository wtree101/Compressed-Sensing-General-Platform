function tensor_T = create_tensor_from_matrix(X, d)
% CREATE_TENSOR_FROM_MATRIX Create fourth-order tensor T = X ⊗ X from matrix X
%
% This function creates a symmetric fourth-order tensor using the tensor
% product (Kronecker product) structure: T = X ⊗ X
%
% Inputs:
%   X - Input matrix of size d x d (typically symmetric)
%   d - Dimension of the matrix (optional, inferred from X if not provided)
%
% Output:
%   tensor_T - Fourth-order tensor of size d x d x d x d
%
% Usage:
%   X = randn(10, 10);
%   T = create_tensor_from_matrix(X);
%   T = create_tensor_from_matrix(X, 10);

    % Handle optional dimension argument
    if nargin < 2
        [d1, d2] = size(X);
        if d1 ~= d2
            error('Matrix X must be square. Got size [%d, %d]', d1, d2);
        end
        d = d1;
    end
    
    % Validate input dimensions
    [d1, d2] = size(X);
    if d1 ~= d || d2 ~= d
        error('Matrix X size [%d, %d] does not match expected dimension d=%d', d1, d2, d);
    end
    
    % Create fourth-order tensor T = X ⊗ X
    % Flatten X to vector and compute outer product
    n = d * d;
    X_vec = reshape(X, n, 1);
    tensor_flat = X_vec * X_vec';  % n x n matrix (tensor product)
    
    % Reshape to 4D tensor
    tensor_T = reshape(tensor_flat, [d, d, d, d]);
end
