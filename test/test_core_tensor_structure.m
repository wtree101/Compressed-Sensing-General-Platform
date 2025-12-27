%% Test Core Tensor Structure from Spectral Initialization
% This test verifies Proposition: Structure of the Core Tensor
%
% Specifically, it tests:
%   (i)   G_mat = (U2' ⊗ U1') * H_mat * (U4' ⊗ U3')'
%   (ii)  If U1=U3, U2=U4, and H_mat is symmetric, then G_mat is symmetric
%   (iii) Rank-1 approximation structure preservation
%
% Test Date: 2025-12-23

clear; clc;

fprintf('=== Test: Core Tensor Structure from Spectral Initialization ===\n\n');

%% Test Parameters
d1 = 10;             % First dimension
d2 = 10;             % Second dimension
r = 3;               % Tucker rank
test_symmetric = true; % Test symmetric case (U1=U3, U2=U4)

rng(42);  % For reproducibility

fprintf('Test Configuration:\n');
fprintf('  Dimensions: d1=%d, d2=%d\n', d1, d2);
fprintf('  Tucker rank: r=%d\n', r);
fprintf('  Test symmetric case: %s\n\n', mat2str(test_symmetric));

%% Generate Test Tensor H (4th-order)
fprintf('=== Step 1: Generate Test Tensor H ===\n');

% Create a random 4th-order tensor H
H_tensor = randn(d1, d2, d1, d2);

% Optional: make it more structured (rank-1 tensor for testing)
if test_symmetric
    % Create rank-1 tensor: H = v ⊗ v where v = vec(X)
    X_test = randn(d1, d2);
    X_test = (X_test + X_test') / 2;  % Symmetric
    v = X_test(:);
    H_mat = v * v';
    H_tensor = reshape(H_mat, [d1, d2, d1, d2]);
    fprintf('Created rank-1 symmetric tensor\n');
else
    H_mat = reshape(H_tensor, [d1*d2, d1*d2]);
end

fprintf('  H tensor size: %dx%dx%dx%d\n', d1, d2, d1, d2);
fprintf('  H tensor norm: %.6f\n', norm(H_tensor(:)));

%% Generate Factor Matrices
fprintf('\n=== Step 2: Generate Factor Matrices ===\n');

if test_symmetric
    % Symmetric case: U1 = U3, U2 = U4
    U1 = orth(randn(d1, r));
    U2 = orth(randn(d2, r));
    U3 = U1;  % Same as U1
    U4 = U2;  % Same as U2
    fprintf('Symmetric case: U1=U3, U2=U4\n');
else
    % General case: all different
    U1 = orth(randn(d1, r));
    U2 = orth(randn(d2, r));
    U3 = orth(randn(d1, r));
    U4 = orth(randn(d2, r));
    fprintf('General case: all factor matrices independent\n');
end

fprintf('  U1: %dx%d\n', size(U1, 1), size(U1, 2));
fprintf('  U2: %dx%d\n', size(U2, 1), size(U2, 2));
fprintf('  U3: %dx%d\n', size(U3, 1), size(U3, 2));
fprintf('  U4: %dx%d\n', size(U4, 1), size(U4, 2));

%% Compute Core Tensor G using tensor_mode_product
fprintf('\n=== Step 3: Compute Core Tensor G ===\n');
fprintf('Computing G = H ×₁ U1^T ×₂ U2^T ×₃ U3^T ×₄ U4^T\n');

tic;
G_tensor = H_tensor;
G_tensor = tensor_mode_product(G_tensor, U1', 1);
G_tensor = tensor_mode_product(G_tensor, U2', 2);
G_tensor = tensor_mode_product(G_tensor, U3', 3);
G_tensor = tensor_mode_product(G_tensor, U4', 4);
elapsed_time = toc;

fprintf('  Core tensor G size: %dx%dx%dx%d\n', size(G_tensor, 1), size(G_tensor, 2), ...
        size(G_tensor, 3), size(G_tensor, 4));
fprintf('  Core tensor norm: %.6f\n', norm(G_tensor(:)));
fprintf('  Computation time: %.4f seconds\n', elapsed_time);

%% Matricize Tensors
fprintf('\n=== Step 4: Matricize Tensors ===\n');

% H_mat: already computed above, or compute now
H_mat = reshape(H_tensor, [d1*d2, d1*d2]);
fprintf('  H_mat: %dx%d matrix\n', size(H_mat, 1), size(H_mat, 2));
fprintf('  H_mat norm: %.6f\n', norm(H_mat, 'fro'));

% Check if H_mat is symmetric
H_mat_symm_error = norm(H_mat - H_mat', 'fro');
fprintf('  ||H_mat - H_mat^T||_F = %.6e\n', H_mat_symm_error);
if H_mat_symm_error < 1e-10
    fprintf('  ✓ H_mat is symmetric\n');
else
    fprintf('  ✗ H_mat is NOT symmetric\n');
end

% G_mat: matricize core tensor
G_mat = reshape(G_tensor, [r*r, r*r]);
fprintf('\n  G_mat: %dx%d matrix\n', size(G_mat, 1), size(G_mat, 2));
fprintf('  G_mat norm: %.6f\n', norm(G_mat, 'fro'));

%% Test Proposition (i): G_mat = (U2' ⊗ U1') * H_mat * (U4' ⊗ U3')'
fprintf('\n=== Test (i): Matrix Representation Formula ===\n');
fprintf('Testing: G_mat = (U2^T ⊗ U1^T) * H_mat * (U4^T ⊗ U3^T)^T\n\n');

% Compute Kronecker products
fprintf('Step (i.1): Computing Kronecker products\n');
U21 = kron(U2', U1');  % (r × d2) ⊗ (r × d1) = (r² × d1d2)
U43 = kron(U4', U3');  % (r × d2) ⊗ (r × d1) = (r² × d1d2)

fprintf('  U2^T ⊗ U1^T: %dx%d\n', size(U21, 1), size(U21, 2));
fprintf('  U4^T ⊗ U3^T: %dx%d\n', size(U43, 1), size(U43, 2));

% Compute G_mat via matrix formula
fprintf('\nStep (i.2): Computing G_mat via formula\n');
G_mat_formula = U21 * H_mat * U43';

fprintf('  G_mat (formula): %dx%d\n', size(G_mat_formula, 1), size(G_mat_formula, 2));
fprintf('  G_mat (formula) norm: %.6f\n', norm(G_mat_formula, 'fro'));

% Compare with G_mat from tensor_mode_product
fprintf('\nStep (i.3): Comparing results\n');
diff_G = norm(G_mat - G_mat_formula, 'fro');
relative_diff = diff_G / max(norm(G_mat, 'fro'), 1e-10);

fprintf('  ||G_mat - G_mat_formula||_F = %.6e\n', diff_G);
fprintf('  Relative difference: %.6e\n', relative_diff);

if relative_diff < 1e-10
    fprintf('  ✓ TEST (i) PASSED: Formula is correct\n');
else
    fprintf('  ✗ TEST (i) FAILED: Formula does not match (relative diff: %.2e)\n', relative_diff);
end

%% Test Proposition (ii): Symmetry Preservation
fprintf('\n=== Test (ii): Symmetry Preservation ===\n');

if test_symmetric
    fprintf('Testing: If U1=U3, U2=U4, and H_mat is symmetric, then G_mat is symmetric\n\n');
    
    % Check symmetry of G_mat
    G_mat_symm_error = norm(G_mat - G_mat', 'fro');
    G_mat_relative_asymm = G_mat_symm_error / norm(G_mat, 'fro');
    
    fprintf('  ||G_mat - G_mat^T||_F = %.6e\n', G_mat_symm_error);
    fprintf('  Relative asymmetry: %.6e\n', G_mat_relative_asymm);
    
    if G_mat_relative_asymm < 1e-10
        fprintf('  ✓ TEST (ii) PASSED: G_mat is symmetric\n');
    else
        fprintf('  ✗ TEST (ii) FAILED: G_mat is NOT symmetric (relative asymm: %.2e)\n', ...
                G_mat_relative_asymm);
    end
else
    fprintf('Skipping Test (ii): not in symmetric case\n');
end

%% Test Proposition (iii): Rank-1 Structure Preservation
fprintf('\n=== Test (iii): Rank-1 Structure Preservation ===\n');

if test_symmetric && H_mat_symm_error < 1e-10
    fprintf('Testing: H_mat ≈ λv·v^T => G_mat ≈ λq·q^T where q = (U2^T ⊗ U1^T)v\n\n');
    
    % Extract leading eigenpair of H_mat
    fprintf('Step (iii.1): Computing leading eigenpair of H_mat\n');
    [V_H, D_H] = eig(H_mat);
    [lambda_H, idx_H] = max(abs(diag(D_H)));
    v_H = V_H(:, idx_H);
    lambda_H = D_H(idx_H, idx_H);
    
    fprintf('  Leading eigenvalue of H_mat: λ = %.6f\n', lambda_H);
    fprintf('  Leading eigenvector norm: ||v|| = %.6f\n', norm(v_H));
    
    % Compute q = (U2' ⊗ U1') * v
    fprintf('\nStep (iii.2): Computing q = (U2^T ⊗ U1^T) * v\n');
    q = U21 * v_H;
    fprintf('  q vector size: %d\n', length(q));
    fprintf('  q vector norm: %.6f\n', norm(q));
    
    % Form rank-1 approximation of G_mat
    fprintf('\nStep (iii.3): Forming rank-1 approximation G_mat ≈ λ·q·q^T\n');
    G_mat_rank1_expected = lambda_H * (q * q');
    
    % Compare with actual G_mat
    diff_rank1 = norm(G_mat - G_mat_rank1_expected, 'fro');
    relative_diff_rank1 = diff_rank1 / norm(G_mat, 'fro');
    
    fprintf('  ||G_mat - λqq^T||_F = %.6e\n', diff_rank1);
    fprintf('  Relative difference: %.6e\n', relative_diff_rank1);
    
    % Also check leading eigenpair of G_mat
    fprintf('\nStep (iii.4): Verifying eigenpair of G_mat\n');
    [V_G, D_G] = eig(G_mat);
    [lambda_G, idx_G] = max(abs(diag(D_G)));
    v_G = V_G(:, idx_G);
    lambda_G = D_G(idx_G, idx_G);
    
    fprintf('  Leading eigenvalue of G_mat: %.6f\n', lambda_G);
    fprintf('  Expected: %.6f\n', lambda_H);
    fprintf('  Eigenvalue difference: %.6e\n', abs(lambda_G - lambda_H));
    
    % Compare eigenvectors (up to sign)
    v_G_normalized = v_G / norm(v_G);
    q_normalized = q / norm(q);
    eigvec_diff = min(norm(v_G_normalized - q_normalized), ...
                      norm(v_G_normalized + q_normalized));
    fprintf('  Leading eigenvector of G_mat vs q (up to sign): %.6e\n', eigvec_diff);
    
    if relative_diff_rank1 < 1e-6 && eigvec_diff < 1e-6
        fprintf('  ✓ TEST (iii) PASSED: Rank-1 structure preserved\n');
    else
        fprintf('  ~ TEST (iii) WARNING: Rank-1 structure approximate (diff: %.2e)\n', ...
                relative_diff_rank1);
    end
else
    fprintf('Skipping Test (iii): H_mat is not symmetric or not in symmetric case\n');
end

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('Test (i) - Matrix representation: ');
if relative_diff < 1e-10
    fprintf('✓ PASSED\n');
else
    fprintf('✗ FAILED (diff: %.2e)\n', relative_diff);
end

if test_symmetric
    fprintf('Test (ii) - Symmetry preservation: ');
    if G_mat_symm_error < 1e-10
        fprintf('✓ PASSED\n');
    else
        fprintf('✗ FAILED (asymm: %.2e)\n', G_mat_symm_error);
    end
    
    if H_mat_symm_error < 1e-10
        fprintf('Test (iii) - Rank-1 structure: ');
        if exist('relative_diff_rank1', 'var') && relative_diff_rank1 < 1e-6
            fprintf('✓ PASSED\n');
        else
            fprintf('~ APPROXIMATE (diff: %.2e)\n', relative_diff_rank1);
        end
    end
end

fprintf('\n=== Test Complete ===\n');

%% Helper Function
function T_out = tensor_mode_product(T, M, mode)
    % TENSOR_MODE_PRODUCT n-mode product of tensor T with matrix M
    % T_out = T ×_mode M
    
    sz = size(T);
    k = size(M, 1);
    
    % Permute so mode is first
    order = 1:max(ndims(T), mode);
    order([1, mode]) = [mode, 1];
    T_perm = permute(T, order);
    
    % Reshape to matrix and multiply
    T_mat = reshape(T_perm, sz(mode), []);
    T_out_mat = M * T_mat;
    
    % Reshape back
    sz_out = sz;
    sz_out(mode) = k;
    sz_out_perm = sz_out(order);
    T_out_perm = reshape(T_out_mat, sz_out_perm);
    
    % Permute back
    T_out = ipermute(T_out_perm, order);
end

