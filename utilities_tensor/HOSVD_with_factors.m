function [B, U_factors] = HOSVD_with_factors(tensor, rank)
% HOSVD_WITH_FACTORS Higher-Order SVD with factor matrices returned
%
% Performs HOSVD decomposition and returns both the projected tensor
% and the factor matrices used for projection.
%
% Inputs:
%   tensor - Input tensor (arbitrary order)
%   rank   - Vector of ranks for each mode [r1, r2, ..., rN]
%
% Outputs:
%   B         - Projected tensor: B = mul(...mul(mul(tensor, 1, U1*U1'), 2, U2*U2')..., N, UN*UN')
%   U_factors - Cell array of factor matrices {U1, U2, ..., UN}
%               Each Ui is di × ri orthonormal matrix
%
% Example:
%   [T_proj, U_cells] = HOSVD_with_factors(T, [5, 5, 5, 5]);
%
% See also: HOSVD, tensor_to_mat, mul

d = ndims(tensor);
B = tensor;
U_factors = cell(1, d);

% Extract U factors for each mode via SVD
for i = 1:d
    mat = tensor_to_mat(tensor, i);
    [U, ~, ~] = svd(mat, 'econ');
    % Truncate to desired rank
    U_factors{i} = U(:, 1:rank(i));
    
    % Project tensor onto subspace spanned by U
    B = mul(B, i, U_factors{i} * U_factors{i}');
end

end

