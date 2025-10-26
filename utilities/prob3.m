function [A, M_star] = prob3(n, kappa, r, r_star)
% Construct measurement matrices A and target M_star as described

% One-hot vectors
U = eye(n);

% Target matrix M^* = sum_{k=r+1}^{r+r^*} u_k u_k^T
M_star = zeros(n);
for k = r+1:r+r_star
    M_star = M_star + U(:,k) * U(:,k)';
end

% Ordering pi = [1, r+1, 2, 3, ..., r, r+2, r+3, ..., r+r^*]
pi = [1, r+1, 2:r, r+2:r+r_star];

% Initialize A as a cell array of n x n matrices
A = cell(n, n);

% Default: A^{(i,j)} = sqrt(kappa) * u_i u_j^T
for i = 1:n
    for j = 1:n
        A{i,j} = sqrt(kappa) * U(:,i) * U(:,j)';
    end
end

% Special cases for diagonal blocks
% A^{(1,1)}
A{1,1} = sqrt(kappa/(2*r)) * (U(:,1:r) * U(:,1:r)') + sqrt(kappa/(2*r_star)) * (U(:,r+1:r+r_star) * U(:,r+1:r+r_star)');

% A^{(r+1,r+1)}
A{r+1,r+1} = (1/sqrt(2*r)) * (U(:,1:r) * U(:,1:r)') - (1/sqrt(2*r_star)) * (U(:,r+1:r+r_star) * U(:,r+1:r+r_star)');

% For i in {3, ..., r+r^*} (in pi ordering)
for idx = 3:(r+r_star)
    i = pi(idx);
    % Start with u_{pi(i)} u_{pi(i)}^T
    Aii = U(:,i) * U(:,i)';
    % Subtract projections onto previous A^{(j,j)}
    for jdx = 1:idx-1
        j = pi(jdx);
        Ajj = A{j,j};
        proj = (trace(Ajj' * (U(:,i) * U(:,i)')) / norm(Ajj,'fro')^2) * Ajj;
        Aii = Aii - proj;
    end
    A{i,i} = Aii;
end

% Convert A from cell to 3D array for output (optional)
A_mat = zeros(n^2, n^2);
for i = 1:n
    for j = 1:n
        A_mat((i-1)*n + j, :) = reshape(A{i,j}, 1, []);
    end
end
A = A_mat;

end