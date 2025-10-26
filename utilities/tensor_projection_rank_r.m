function T_proj = tensor_projection_rank_r(T, r)
    rank = [r, r, r, r];
    T_proj = HOSVD(T,rank);
end

% function X_proj = project_symmetric_low_rank(X, r)
%     % Project symmetric matrix X to rank-r
%     X = (X + X') / 2;
%     if r <= 0 || r >= size(X, 1)
%         X_proj = X;
%         return;
%     end
%     [V, D] = eig(X);
%     [~, idx] = sort(abs(diag(D)), 'descend');
%     V = V(:, idx);
%     D = D(idx, idx);
%     D_proj = zeros(size(D));
%     D_proj(1:r, 1:r) = D(1:r, 1:r);
%     X_proj = V * D_proj * V';
%     X_proj = (X_proj + X_proj') / 2;
% end



% function X = extract_matrix_from_tensor(T, params)
%     % Extract matrix X from 4D tensor T assuming T ≈ X ⊗ X structure
%     % Method: Matricization and leading singular vectors
%     d = size(T, 1);
%     n = d * d;
%     r = params.r;
%     if isfield(params, 'r')
%         r = params.r;
%     else
%         error('Rank r must be specified in params for matrix extraction.');
%     end
%     % Mode-1,2 matricization: T_{(1,2)} is n x n
%     T_mat = reshape(T, [n, n]);
%     % Check symmetry
%     if norm(T_mat - T_mat', 'fro') < 1e-10
%         fprintf('T_mat is symmetric (||T_mat - T_mat''||_F < 1e-10)\n');
%     else
%         fprintf('T_mat is NOT symmetric (||T_mat - T_mat''||_F = %.2e)\n', norm(T_mat - T_mat', 'fro'));
%     end
%     
%     % Symmetrize
%     T_mat = (T_mat + T_mat') / 2;
%     
%     % Extract leading eigenvector and reshape to matrix
%     [V, D] = eig(T_mat);
%     [~, idx] = sort(abs(diag(D)), 'descend');
%     v = V(:, idx(1)); % Leading eigenvector
%     
%     % Reshape to matrix and symmetrize
%     X = reshape(v * sqrt(abs(D(idx(1), idx(1)))), [d, d]);
%     % Check symmetry of X
%     symmetry_error = norm(X - X', 'fro');
%     fprintf('Extracted X symmetry error: %.2e\n', symmetry_error);
% 
%     X = (X + X') / 2;
%     % Project X to rank-r approximation
%     X = project_symmetric_low_rank(X, r);
% end