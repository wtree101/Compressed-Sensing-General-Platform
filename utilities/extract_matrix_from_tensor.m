function X = extract_matrix_from_tensor(T, params)
% EXTRACT_MATRIX_FROM_TENSOR Extract matrix X from 4th order tensor T
% 
% This function extracts a matrix X from a 4th order tensor T assuming 
% the structure T ≈ X ⊗ X (tensor product form) for symmetric tensor 
% phase retrieval problems.
%
% Inputs:
%   T      - 4th order tensor of size d x d x d x d
%   params - Structure with fields:
%            .r      - Target rank for extracted matrix (required)
%            .method - Extraction method: 'eig', 'svd', 'hosvd' (optional, default: 'eig')
%            .symmetrize - Whether to enforce symmetry (optional, default: true)
%            .verbose    - Print diagnostic info (optional, default: false)
%
% Output:
%   X      - Extracted symmetric matrix of size d x d with rank <= r
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
    if ~isfield(params, 'symmetrize'), params.symmetrize = true; end
    if ~isfield(params, 'verbose'), params.verbose = false; end
    
    % Get tensor dimensions
    if ndims(T) ~= 4
        error('Input tensor T must be 4th order (4 dimensions)');
    end
    
    [d1, d2, d3, d4] = size(T);
    if d1 ~= d2 || d2 ~= d3 || d3 ~= d4
        error('Tensor must be d x d x d x d (currently [%d,%d,%d,%d])', d1, d2, d3, d4);
    end
    d = d1;
    r = params.r;
    
    % Apply method-specific extraction
    switch lower(params.method)
        case 'eig'
            X = extract_eigen_method(T, d, r, params);
        case 'svd'
            X = extract_svd_method(T, d, r, params);
        case 'hosvd'
            X = extract_hosvd_method(T, d, r, params);
        otherwise
            error('Unknown method: %s. Use ''eig'', ''svd'', or ''hosvd''', params.method);
    end


    
    % Check whether X is symmetric
    is_symmetric = norm(X - X', 'fro') < 1e-10;
    if params.verbose
        fprintf('Is X symmetric? %d\n', is_symmetric);
    end
    if params.symmetrize
        X = (X + X') / 2;
        X = project_symmetric_low_rank(X, r);
    end
    
    if params.verbose
        fprintf('Final extracted matrix: norm=%.3f, rank=%d, symmetry_error=%.2e\n', ...
                norm(X,'fro'), rank(X), norm(X-X','fro'));
    end
end

function X = extract_eigen_method(T, d, ~, params)
% Eigendecomposition method (most stable for symmetric tensors)

    n = d * d;
    
    % Mode-(1,2) matricization: reshape T to n x n matrix
    T_mat = reshape(T, [n, n]);
    
    if params.verbose
        fprintf('Mode-(1,2) matricization: %dx%d -> %dx%d\n', d, d, n, n);
        fprintf('T_mat symmetry error: %.2e\n', norm(T_mat - T_mat', 'fro'));
    end
    
    % Symmetrize the matricized tensor
    T_mat = (T_mat + T_mat') / 2;
    
    % Extract leading eigenvector
    [V, D] = eig(T_mat);
    [~, idx] = sort(abs(diag(D)), 'descend');
    v = V(:, idx(1)); % Leading eigenvector
    lambda = D(idx(1), idx(1));
    
    if params.verbose
        fprintf('Leading eigenvalue: %.6f\n', lambda);
    end
    
    % Reshape to matrix form
    X = reshape(v * sqrt(abs(lambda)), [d, d]);
end

function X = extract_svd_method(T, d, ~, params)
% SVD-based method (alternative approach)

    n = d * d;
    
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
    
    % For symmetric case, u1 ≈ v1, so use average
    x_vec = (u1 + v1) / 2 * sqrt(S(1,1));
    X = reshape(x_vec, [d, d]);
end

function X = extract_hosvd_method(T, d, r, params)
% HOSVD preprocessing method (for noisy tensors)

    if params.verbose
        fprintf('HOSVD method: preprocessing with rank [%d,%d,%d,%d]\n', r, r, r, r);
    end
    
    % Apply HOSVD projection first
    rank_vec = [r, r, r, r];
    T_clean = HOSVD(T, rank_vec);
    
    % Then use eigendecomposition on cleaned tensor
    X = extract_eigen_method(T_clean, d, r, params);
end

