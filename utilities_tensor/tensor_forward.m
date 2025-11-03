function y = tensor_forward(T, A_tensor, d)
% TENSOR_FORWARD Forward operator for fourth-order tensor measurements
%
% Computes y = A(T) where A is a linear measurement operator and T is a
% fourth-order tensor. Each measurement is y_i = ⟨A_i, T⟩ where A_i is
% the i-th measurement tensor (stored as flattened row in A_tensor).
%
% Inputs:
%   T        - Fourth-order tensor of size d x d x d x d
%   A_tensor - Measurement matrix of size m x (d^4), where each row 
%              represents a flattened measurement tensor A_i
%   d        - Dimension parameter (tensor is d x d x d x d)
%
% Output:
%   y - Measurement vector of size m x 1
%
% Usage:
%   T = randn(5, 5, 5, 5);
%   A = randn(100, 5^4);  % 100 measurements
%   y = tensor_forward(T, A, 5);

    % Validate inputs
    if nargin < 3
        error('All three arguments (T, A_tensor, d) are required');
    end
    
    % Flatten tensor to vector
    n = d * d;
    T_vec = reshape(T, [n * n, 1]);
    
    % Apply measurement operator
    y = A_tensor * T_vec;
end
