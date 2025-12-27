function X = extract_matrix_from_tensor(T, params)
% EXTRACT_MATRIX_FROM_TENSOR Extract matrix X from 4th order tensor T (Non-Symmetric)
% 
% This function extracts a matrix X from a 4th order tensor T assuming 
% the structure T ≈ X ⊗ X (tensor product form).
% SUPPORTS NON-SYMMETRIC MATRICES (d1 can differ from d2).
%
% Inputs:
%   T      - 4th order tensor of size d1 x d2 x d1 x d2
%   params - Structure with fields:
%            .r      - Target rank for extracted matrix (required)
%            .d1     - Row dimension (optional, inferred from T if not provided)
%            .d2     - Column dimension (optional, inferred from T if not provided)
%            .method - Extraction method: 'eig', 'svd', 'hosvd' (optional, default: 'eig')
%            .verbose    - Print diagnostic info (optional, default: false)
%
% Output:
%   X      - Extracted matrix of size d1 x d2 with rank <= r
%
% Methods:
%   'eig'   - Eigendecomposition of matricized tensor (most stable)
%   'svd'   - SVD-based extraction (alternative approach) 
%   'hosvd' - Higher-order SVD preprocessing (for noisy cases)

    % Input validation
    if nargin < 2 || ~isfield(params, 'r')
        error('params.r (target rank) must be specified');
    end
    
    % Default parameters
    if ~isfield(params, 'method'), params.method = 'eig'; end
    if ~isfield(params, 'verbose'), params.verbose = false; end
    
    % Get tensor dimensions
    if ndims(T) ~= 4
        error('Input tensor T must be 4th order (4 dimensions)');
    end
    
    [d1, d2, d3, d4] = size(T);
    
    % Check tensor structure: T should be d1 x d2 x d1 x d2
    if d1 ~= d3 || d2 ~= d4
        error('Tensor must be d1 x d2 x d1 x d2 (currently [%d,%d,%d,%d])', d1, d2, d3, d4);
    end
    
    % Allow d1, d2 to be provided in params (for verification)
    if isfield(params, 'd1') && params.d1 ~= d1
        warning('params.d1=%d does not match tensor dimension d1=%d', params.d1, d1);
    end
    if isfield(params, 'd2') && params.d2 ~= d2
        warning('params.d2=%d does not match tensor dimension d2=%d', params.d2, d2);
    end
    
    r = params.r;
    
    % Apply method-specific extraction
    switch lower(params.method)
        case 'eig'
            X = extract_eigen_method(T, d1, d2, r, params);
        case 'svd'
            X = extract_svd_method(T, d1, d2, r, params);
        case 'hosvd'
            X = extract_hosvd_method(T, d1, d2, r, params);
        otherwise
            error('Unknown method: %s. Use ''eig'', ''svd'', or ''hosvd''', params.method);
    end

    % NO SYMMETRIZATION - keep original non-symmetric structure
    % Apply low-rank projection if needed
    if rank(X) > r
        [U, S, V] = svd(X);
        X = U(:, 1:r) * S(1:r, 1:r) * V(:, 1:r)';
    end
    
    if params.verbose
        fprintf('Final extracted matrix: norm=%.3f, rank=%d, size=[%d,%d]\n', ...
                norm(X,'fro'), rank(X), size(X,1), size(X,2));
    end
end

function X = extract_eigen_method(T, d1, d2, ~, params)
% Eigendecomposition method (most stable for tensors)
% For non-symmetric case, extracts from matricized tensor

    n = d1 * d2;
    
    % Mode-(1,2) matricization: reshape T to n x n matrix
    T_mat = reshape(T, [n, n]);
    
    if params.verbose
        fprintf('Mode-(1,2) matricization: %dx%d -> %dx%d\n', d1, d2, n, n);
        fprintf('T_mat symmetry check: %.2e\n', norm(T_mat - T_mat', 'fro'));
    end
    
    % For non-symmetric case, symmetrize only for eigendecomposition stability
    % but don't enforce symmetry on final result
    T_mat_sym = (T_mat + T_mat') / 2;
    
    % Extract leading eigenvector
    [V, D] = eig(T_mat_sym);
    [~, idx] = sort(abs(diag(D)), 'descend');
    v = V(:, idx(1)); % Leading eigenvector
    lambda = D(idx(1), idx(1));
    
    if params.verbose
        fprintf('Leading eigenvalue: %.6f\n', lambda);
    end
    
    % Reshape to matrix form (d1 x d2)
    X = reshape(v * sqrt(abs(lambda)), [d1, d2]);
end

function X = extract_svd_method(T, d1, d2, ~, params)
% SVD-based method (alternative approach for non-symmetric case)

    n = d1 * d2;
    
    % Mode-(1,2) matricization
    T_mat = reshape(T, [n, n]);
    
    if params.verbose
        fprintf('SVD method: matricization %dx%d\n', n, n);
    end
    
    % Use SVD for rank-1 approximation
    [U, S, V] = svds(T_mat, 1);
    
    if params.verbose
        fprintf('Leading singular value: %.6f\n', S(1,1));
    end
    
    % Extract matrix from leading singular vectors
    u1 = U(:, 1);
    v1 = V(:, 1);
    
    % For non-symmetric case, average u1 and v1 for stability
    x_vec = (u1 + v1) / 2 * sqrt(S(1,1));
    X = reshape(x_vec, [d1, d2]);
end

function X = extract_hosvd_method(T, d1, d2, r, params)
% HOSVD preprocessing method (for noisy tensors, non-symmetric case)

    if params.verbose
        fprintf('HOSVD method: preprocessing with rank [%d,%d,%d,%d]\n', r, r, r, r);
    end
    
    % Apply HOSVD projection first
    rank_vec = [r, r, r, r];
    T_clean = HOSVD(T, rank_vec);
    
    % Then use eigendecomposition on cleaned tensor
    X = extract_eigen_method(T_clean, d1, d2, r, params);
end

