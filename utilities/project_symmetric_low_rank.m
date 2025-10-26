function X_proj = project_symmetric_low_rank(X, r)
% Project symmetric matrix X to rank-r using existing helper function
% (This matches the existing implementation in tensor_projection_rank_r.m)

    X = (X + X') / 2;  % Ensure symmetry
    if r <= 0 || r >= size(X, 1)
        X_proj = X;
        return;
    end
    
    [V, D] = eig(X);
    [~, idx] = sort(abs(diag(D)), 'descend');
    V = V(:, idx);
    D = D(idx, idx);
    
    % Keep only top r eigenvalues
    D_proj = zeros(size(D));
    D_proj(1:r, 1:r) = D(1:r, 1:r);
    X_proj = V * D_proj * V';
    X_proj = (X_proj + X_proj') / 2;  % Ensure final symmetry
end
