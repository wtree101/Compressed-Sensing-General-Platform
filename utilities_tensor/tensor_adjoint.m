function T = tensor_adjoint(z, A_tensor, d)
% TENSOR_ADJOINT Adjoint operator for fourth-order tensor measurements
%
% Computes T = A^*(z) where A^* is the adjoint of the measurement operator
% and z is a measurement vector. This is the transpose operation that maps
% measurements back to tensor space.
%
% Inputs:
%   z        - Measurement vector of size m x 1
%   A_tensor - Measurement matrix of size m x (d^4), where each row 
%              represents a flattened measurement tensor A_i
%   d        - Dimension parameter (output tensor is d x d x d x d)
%
% Output:
%   T - Fourth-order tensor of size d x d x d x d
%
% Usage:
%   z = randn(100, 1);  % 100 measurements
%   A = randn(100, 5^4);
%   T = tensor_adjoint(z, A, 5);

    % Validate inputs
    if nargin < 3
        error('All three arguments (z, A_tensor, d) are required');
    end
    
    % Apply adjoint measurement operator
    T_vec = A_tensor' * z;
    
    % Reshape back to 4D tensor
    T = reshape(T_vec, [d, d, d, d]);
end
