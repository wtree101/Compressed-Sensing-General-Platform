%% Test Tucker Tensor for Non-Symmetric Case
% This test verifies Tucker tensor operations when factor matrices are NOT all identical
% i.e., U{1}, U{2}, U{3}, U{4} may be different
%
% Test scenarios:
%   1. General case: All factors different
%   2. Partial symmetry: U1=U3, U2=U4 (common in tensor phase retrieval)
%   3. Formula verification: G_mat = (U2' ⊗ U1') * H_mat * (U4' ⊗ U3')'
%
% Test Date: 2025-12-23

clear; clc;

fprintf('=== Test: Tucker Tensor Non-Symmetric Case ===\n\n');

%% Test Parameters
d1 = 20;             % First dimension (rows)
d2 = 30;             % Second dimension (cols)
r = 1;               % Tucker rank
test_case = 'partial';  % 'full' = all different, 'partial' = U1=U3, U2=U4

%rng(42);  % For reproducibility

fprintf('Test Configuration:\n');
fprintf('  Dimension d1 (rows): %d\n', d1);
fprintf('  Dimension d2 (cols): %d\n', d2);
fprintf('  Tucker rank r: %d\n', r);
fprintf('  Test case: %s symmetry\n\n', test_case);

%% Generate Ground Truth and Measurements
fprintf('=== Step 1: Generate Ground Truth and Measurements ===\n');

% Create ground truth matrix X_true
X_true = abs(randn(d1, r))*abs(randn(r,d2));
X_true = X_true / norm(X_true, 'fro');
fprintf('Ground truth X_true: %dx%d (non-square, non-symmetric), norm=%.6f\n', ...
        d1, d2, norm(X_true, 'fro'));

% Number of measurements
m = 1000;
fprintf('Number of measurements: %d\n', m);

% Generate random measurement matrices
fprintf('Generating %d random measurement matrices...\n', m);
A_cells = cell(m, 1);
for i = 1:m
    Ai = randn(d1, d2) ;
    A_cells{i} = Ai;
end

% Compute measurements: y_i = |<Ai, X_true>|^2
y = zeros(m, 1);
for i = 1:m
    y(i) = abs(sum(sum(A_cells{i} .* X_true)))^2 / sqrt(m);
end
fprintf('Measurements computed: y ∈ R^%d\n', m);

%% Generate Test Tensor H via Spectral Initialization
fprintf('\n=== Step 2: Generate Tensor H via Spectral Initialization ===\n');
fprintf('Using spectral operator: H = sum_i y_i * (Ai ⊗ Ai)\n');

% Create spectral tensor: H = sum_i y_i * (Ai ⊗ Ai)
% This is what initialize_spectral does internally
H_tensor = zeros(d1, d2, d1, d2);
y_spectral = y ;  % Preprocessing as in spectral init

for i = 1:m
    Ai = A_cells{i};
    % Form Ai ⊗ Ai
    for i1 = 1:d1
        for i2 = 1:d2
            for j1 = 1:d1
                for j2 = 1:d2
                    H_tensor(i1, i2, j1, j2) = H_tensor(i1, i2, j1, j2) + ...
                        y_spectral(i) * Ai(i1, i2) * Ai(j1, j2) / sqrt(m);
                end
            end
        end
    end
end

fprintf('  Spectral tensor H generated: size %dx%dx%dx%d\n', ...
        size(H_tensor, 1), size(H_tensor, 2), size(H_tensor, 3), size(H_tensor, 4));
fprintf('  H tensor norm: %.6f\n', norm(H_tensor(:)));

% Theoretical expectation for rank-1 case
fprintf('\nTheoretical structure:\n');
fprintf('  If measurements are perfect and m is large:\n');
fprintf('  H ≈ λ * (X_true ⊗ X_true) for some λ > 0\n');
fprintf('  This means H should have approximate rank-1 structure\n');

%% Extract Factor Matrices via HOSVD
fprintf('\n=== Step 3: Extract Factor Matrices via HOSVD ===\n');
fprintf('Performing HOSVD on spectral tensor H to extract factor matrices...\n');

% Apply HOSVD to extract factor matrices
[T_tensor, U_cell] = HOSVD_with_factors(H_tensor, [r, r, r, r]);

U1 = U_cell{1};
U2 = U_cell{2};
U3 = U_cell{3};
U4 = U_cell{4};

fprintf('  U1: %dx%d, condition number: %.2e\n', size(U1, 1), size(U1, 2), cond(U1));
fprintf('  U2: %dx%d, condition number: %.2e\n', size(U2, 1), size(U2, 2), cond(U2));
fprintf('  U3: %dx%d, condition number: %.2e\n', size(U3, 1), size(U3, 2), cond(U3));
fprintf('  U4: %dx%d, condition number: %.2e\n', size(U4, 1), size(U4, 2), cond(U4));

% Verify symmetry relationships
diff_U13 = norm(U1 - U3, 'fro');
diff_U24 = norm(U2 - U4, 'fro');

fprintf('\nFactor matrix differences (testing for partial symmetry):\n');
fprintf('  ||U1 - U3||_F = %.6e\n', diff_U13);
fprintf('  ||U2 - U4||_F = %.6e\n', diff_U24);

fprintf('\nExpected behavior:\n');
fprintf('  For spectral init H = sum_i y_i * (Ai ⊗ Ai) from phase retrieval:\n');
fprintf('  Should have U1≈U3, U2≈U4 (partial symmetry)\n');
fprintf('  Current test_case: %s\n', test_case);

if diff_U13 < 0.1 && diff_U24 < 0.1
    fprintf('  ✓ Partial symmetry detected (differences < 0.1)\n');
elseif diff_U13 < 0.5 && diff_U24 < 0.5
    fprintf('  ~ Approximate partial symmetry (differences < 0.5)\n');
    fprintf('    This can happen with finite m or numerical precision\n');
else
    fprintf('  ~ Weak or no partial symmetry detected\n');
    fprintf('    Reasons: finite m=%d, noise, or test_case setting\n', m);
end

%% Compute Core Tensor via tensor_mode_product
fprintf('\n=== Step 4: Compute Core Tensor G ===\n');
fprintf('Computing: G = H ×₁ U1^T ×₂ U2^T ×₃ U3^T ×₄ U4^T\n');

tic;
G_tensor = H_tensor;
G_tensor = tensor_mode_product(G_tensor, U1', 1);
G_tensor = tensor_mode_product(G_tensor, U2', 2);
G_tensor = tensor_mode_product(G_tensor, U3', 3);
G_tensor = tensor_mode_product(G_tensor, U4', 4);
time_tensor = toc;


fprintf('  Core tensor G computed: %dx%dx%dx%d\n', size(G_tensor, 1), size(G_tensor, 2), ...
        size(G_tensor, 3), size(G_tensor, 4));
fprintf('  G tensor norm: %.6f\n', norm(G_tensor(:)));
fprintf('  Computation time: %.4f seconds\n', time_tensor);

%% Test Core Tensor Structure
fprintf('\n=== Step 4b: Test Core Tensor Diagonal Structure ===\n');
fprintf('Testing whether G has the structure: G_{ijkl} = σ_i*σ_k if i=j and k=l, 0 otherwise\n\n');

% Extract diagonal elements
fprintf('Extracting diagonal elements from core tensor...\n');
G_diag_values = zeros(r, r);
for k = 1:r
    for l = 1:r
        G_diag_values(k, l) = G_tensor(k, k, l, l);
    end
end

fprintf('Diagonal structure G(k,k,l,l):\n');
disp(G_diag_values);

% Check if core is diagonal (only non-zero on i=j, k=l positions, i.e., G(i,i,k,k))
G_diag_tensor = zeros(r, r, r, r);
for k = 1:r
    for l = 1:r
        G_diag_tensor(k, k, l, l) = G_tensor(k, k, l, l);
    end
end

off_diagonal_norm = norm(G_tensor(:) - G_diag_tensor(:));
total_norm = norm(G_tensor(:));
relative_off_diag = off_diagonal_norm / max(total_norm, 1e-10);

fprintf('\nDiagonal structure analysis:\n');
fprintf('  ||G - G_diag||_F = %.6e\n', off_diagonal_norm);
fprintf('  ||G||_F = %.6e\n', total_norm);
fprintf('  Relative off-diagonal energy: %.6e\n', relative_off_diag);

if relative_off_diag < 1e-6
    fprintf('  ✓ Core tensor is DIAGONAL (i=j, k=l structure, i.e., G(i,i,k,k))\n');
    
    % Try to factorize G_diag_values = diag(σ) * diag(σ)'
    fprintf('\nTesting factorization: G(k,k,l,l) = σ_k * σ_l\n');
    
    % Use SVD to find σ
    [U_svd, S_svd, V_svd] = svd(G_diag_values);
    
    % For rank-1 structure, first singular value dominates
    singular_values = diag(S_svd);
    fprintf('  Singular values of G_diag_values:\n');
    for i = 1:min(r, 5)
        fprintf('    σ_%d = %.6e\n', i, singular_values(i));
    end
    
    % Check rank-1 structure
    rank1_approx = singular_values(1) * (U_svd(:,1) * V_svd(:,1)');
    rank1_error = norm(G_diag_values - rank1_approx, 'fro') / norm(G_diag_values, 'fro');
    
    fprintf('\n  Rank-1 approximation error: %.6e\n', rank1_error);
    
    if rank1_error < 1e-3
        fprintf('  ✓ G_diag_values ≈ σ ⊗ σ (rank-1 structure)\n');
        
        % Extract σ vector
        sigma_vec = sqrt(abs(singular_values(1))) * U_svd(:,1);
        
        fprintf('  Extracted σ vector:\n');
        for i = 1:r
            fprintf('    σ_%d = %.6f\n', i, sigma_vec(i));
        end
        
        % Verify: G(k,k,l,l) ≈ σ_k * σ_l
        G_reconstructed = sigma_vec * sigma_vec';
        recon_error = norm(G_diag_values - G_reconstructed, 'fro') / norm(G_diag_values, 'fro');
        fprintf('\n  Verification: ||G_diag - σ*σ^T||_F / ||G_diag||_F = %.6e\n', recon_error);
        
        if recon_error < 1e-3
            fprintf('  ✓✓ Core tensor has EXACT diagonal structure: G_{iikk} = σ_i*σ_k\n');
        else
            fprintf('  ~ Core tensor has approximate diagonal structure\n');
        end
    else
        fprintf('  ~ G_diag_values is NOT rank-1 (σ ⊗ σ structure not satisfied)\n');
        fprintf('    This can happen with noise or multiple dominant components\n');
    end
    
elseif relative_off_diag < 0.1
    fprintf('  ~ Core tensor is APPROXIMATELY diagonal (%.2f%% off-diagonal)\n', ...
            relative_off_diag * 100);
    fprintf('    This is expected for finite measurements m=%d\n', m);
else
    fprintf('  ✗ Core tensor is NOT diagonal (%.2f%% off-diagonal)\n', ...
            relative_off_diag * 100);
    fprintf('    The diagonal structure G_{ijkl} = σ_i*σ_k if i=j, k=l is NOT satisfied\n');
end

%% Test Formula: G_mat = (U2' ⊗ U1') * H_mat * (U4' ⊗ U3')'
fprintf('\n=== Step 5: Test Matrix Representation Formula ===\n');
fprintf('Testing: G_mat = (U2^T ⊗ U1^T) * H_mat * (U4^T ⊗ U3^T)^T\n\n');

% Matricize tensors
% H: d1 × d2 × d1 × d2 → H_mat: (d1*d2) × (d1*d2)
H_mat = reshape(H_tensor, [d1*d2, d1*d2]);
G_mat_tensor = reshape(G_tensor, [r*r, r*r]);

fprintf('Step 1: Matricize tensors\n');
fprintf('  H_mat: %dx%d\n', size(H_mat, 1), size(H_mat, 2));
fprintf('  G_mat (from tensor): %dx%d\n', size(G_mat_tensor, 1), size(G_mat_tensor, 2));

% Compute Kronecker products
fprintf('\nStep 2: Compute Kronecker products\n');
% U2': r × d2, U1': r × d1 → U21: r² × (d1*d2)
U21 = kron(U2', U1');
% U4': r × d2, U3': r × d1 → U43: r² × (d1*d2)
U43 = kron(U4', U3');

fprintf('  U2^T ⊗ U1^T: %dx%d\n', size(U21, 1), size(U21, 2));
fprintf('  U4^T ⊗ U3^T: %dx%d\n', size(U43, 1), size(U43, 2));

% Compute G_mat via formula
fprintf('\nStep 3: Compute G_mat via formula\n');
tic;
G_mat_formula = U21 * H_mat * U43';
time_formula = toc;

fprintf('  G_mat (formula): %dx%d\n', size(G_mat_formula, 1), size(G_mat_formula, 2));
fprintf('  Computation time: %.4f seconds\n', time_formula);

% Compare results
fprintf('\nStep 4: Compare results\n');
diff_G = norm(G_mat_tensor - G_mat_formula, 'fro');
rel_diff_G = diff_G / max(norm(G_mat_tensor, 'fro'), 1e-10);

fprintf('  ||G_mat_tensor - G_mat_formula||_F = %.6e\n', diff_G);
fprintf('  Relative difference: %.6e\n', rel_diff_G);

if rel_diff_G < 1e-10
    fprintf('  ✓ Formula verification PASSED (diff < 1e-10)\n');
elseif rel_diff_G < 1e-6
    fprintf('  ✓ Formula verification PASSED (diff < 1e-6)\n');
else
    fprintf('  ✗ Formula verification FAILED (diff = %.2e)\n', rel_diff_G);
end

%% Check Symmetry Properties
fprintf('\n=== Step 6: Check Symmetry Properties ===\n');

% Check if H_mat is symmetric
H_mat_symm_error = norm(H_mat - H_mat', 'fro') / norm(H_mat, 'fro');
fprintf('H_mat symmetry:\n');
fprintf('  ||H_mat - H_mat^T||_F / ||H_mat||_F = %.6e\n', H_mat_symm_error);
if H_mat_symm_error < 1e-10
    fprintf('  ✓ H_mat is symmetric\n');
else
    fprintf('  ~ H_mat is NOT symmetric (expected for non-square X)\n');
end

% Check if G_mat is symmetric
G_mat_symm_error = norm(G_mat_tensor - G_mat_tensor', 'fro') / norm(G_mat_tensor, 'fro');
fprintf('\nG_mat symmetry:\n');
fprintf('  ||G_mat - G_mat^T||_F / ||G_mat||_F = %.6e\n', G_mat_symm_error);
if G_mat_symm_error < 1e-10
    fprintf('  ✓ G_mat is symmetric\n');
else
    fprintf('  ~ G_mat is NOT symmetric\n');
end

% Theoretical expectation
fprintf('\nTheoretical expectation:\n');
if d1 == d2
    fprintf('  With d1=d2 and if X were symmetric:\n');
    fprintf('    H_mat would be symmetric → G_mat would be symmetric\n');
    if strcmp(test_case, 'partial') && H_mat_symm_error < 1e-10
        fprintf('  With U1=U3, U2=U4, and H_mat symmetric:\n');
        fprintf('  G_mat SHOULD be symmetric\n');
        if G_mat_symm_error < 1e-6
            fprintf('  ✓ Theory confirmed\n');
        else
            fprintf('  ✗ Unexpected asymmetry (%.2e)\n', G_mat_symm_error);
        end
    end
else
    fprintf('  With d1≠d2 (%d≠%d):\n', d1, d2);
    fprintf('    X is %dx%d (non-square) → H_mat and G_mat NOT symmetric\n', d1, d2);
    fprintf('    This is expected and correct\n');
end

%% Matrix Extraction for Partial Symmetry Case
if strcmp(test_case, 'partial')
    fprintf('\n=== Step 7: Matrix Extraction (Partial Symmetry Case) ===\n');
    fprintf('Testing extraction methods for U1=U3, U2=U4 case\n\n');
    
    % Method 1 (OLD): Direct eigendecomposition of T_mat (d1*d2 × d1*d2)
    fprintf('Method 1 (OLD): Direct eigendecomposition of T_mat\n');
    fprintf('  (Large matrix: %dx%d)\n', d1*d2, d1*d2);
    T_mat = reshape(T_tensor,[d1*d2,d1*d2]);
    tic;
    [V_H, D_H] = eig(T_mat);
    [lambda_H, idx_H] = max(abs(diag(D_H)));
    v_H = V_H(:, idx_H);
    time_method1 = toc;
    
    X_method1 = reshape(v_H * sqrt(abs(lambda_H)), [d1, d2]);
    X_method1 = X_method1 / norm(X_method1, 'fro');
    
    error_method1 = min(norm(X_method1 - X_true, 'fro'), ...
                        norm(X_method1 + X_true, 'fro'));
    rank_method1 = rank(X_method1, 1e-10);
    fprintf('  X_method1: %dx%d (no symmetrization)\n', d1, d2);
    fprintf('  Rank of X_method1: %d\n', rank_method1);
    fprintf('  Reconstruction error: %.6e\n', error_method1);
    fprintf('  Time: %.4f seconds\n', time_method1);
    
    % Method 2 (NEW DEFAULT): Via G_mat eigendecomposition with sign rectification
    fprintf('\nMethod 2 (NEW DEFAULT): Core eigendecomposition + sign rectification\n');
    fprintf('  (Small matrix: %dx%d - much faster!)\n', r*r, r*r);
    tic;
    
    % Step 1: Rectify sign ambiguity to enforce U1≈U3, U2≈U4
    U1_rect = U1;
    U2_rect = U2;
    U3_rect = U3;
    U4_rect = U4;
    
    for i = 1:r
        % Align U3(:,i) with U1(:,i)
        if dot(U1(:,i), U3(:,i)) < 0
            U3_rect(:,i) = -U3(:,i);
        end
        % Align U4(:,i) with U2(:,i)
        if dot(U2(:,i), U4(:,i)) < 0
            U4_rect(:,i) = -U4(:,i);
        end
    end
    
    fprintf('  Step 1: Sign rectification complete\n');
    fprintf('    ||U1 - U3_rect||_F = %.6e (after sign flip)\n', norm(U1 - U3_rect, 'fro'));
    fprintf('    ||U2 - U4_rect||_F = %.6e (after sign flip)\n', norm(U2 - U4_rect, 'fro'));
    
    % Recompute core with rectified factors
    % We need sign correction matrices
    S3 = diag(sign(diag(U3_rect' * U3)));
    S4 = diag(sign(diag(U4_rect' * U4)));
    
    G_rect = G_tensor;
    for c = 1:r
        for dd = 1:r
            G_rect(:, :, c, dd) = G_tensor(:, :, c, dd) * S3(c,c) * S4(dd,dd);
        end
    end
    
    % Step 2: Form G_mat
    G_mat_rect = reshape(G_rect, [r*r, r*r]);
    
    % Check symmetry
    G_mat_symm_error_rect = norm(G_mat_rect - G_mat_rect', 'fro') / max(norm(G_mat_rect, 'fro'), 1e-10);
    fprintf('  Step 2: G_mat formed, symmetry error: %.6e\n', G_mat_symm_error_rect);
    
    if G_mat_symm_error_rect > 1e-6
        fprintf('    Symmetrizing G_mat...\n');
        G_mat_rect = (G_mat_rect + G_mat_rect') / 2;
    end
    
    % Step 3: Eigendecompose small G_mat (r² × r²)
    [V_G, D_G] = eig(G_mat_rect);
    [lambda_G, idx_G] = max(abs(diag(D_G)));
    q_G = V_G(:, idx_G);
    
    fprintf('  Step 3: Eigendecomposition of G_mat complete\n');
    fprintf('    Leading eigenvalue: %.6f\n', lambda_G);
    
    % Step 4: Reconstruct v = (U2 ⊗ U1) * q
    U_kron = kron(U2, U1);  % (d1*d2) × r²
    v_reconstructed = U_kron * q_G;
    
    % Step 5: Reshape to get X
    X_method2 = reshape(v_reconstructed * sqrt(abs(lambda_G)), [d1, d2]);
    X_method2 = X_method2 / norm(X_method2, 'fro');
    
    time_method2 = toc;
    
    error_method2 = min(norm(X_method2 - X_true, 'fro'), ...
                        norm(X_method2 + X_true, 'fro'));
    rank_method2 = rank(X_method2, 1e-10);
    fprintf('  X_method2: %dx%d (no symmetrization)\n', d1, d2);
    fprintf('  Rank of X_method2: %d\n', rank_method2);
    fprintf('  Reconstruction error: %.6e\n', error_method2);
    fprintf('  Time: %.4f seconds\n', time_method2);
    
    % Method 3 (PROJECTED POWER): Projected power iteration on diagonal support
    fprintf('\nMethod 3 (PROJECTED POWER): Diagonal-projected power iteration\n');
    fprintf('  (Exploits diagonal structure: G(i,j,k,l) ≈ 0 unless i=j AND k=l)\n');
    tic;
    
    % Use the same G_mat_rect from Method 2
    % Initialize on diagonal support
    q_proj = zeros(r*r, 1);
    for k = 1:r
        idx = (k-1)*r + k;
        q_proj(idx) = 1/sqrt(r);
    end
    
    max_iter_power = 50;
    tol_power = 1e-6;
    
    fprintf('  Starting projected power iteration (max_iter=%d)...\n', max_iter_power);
    for iter = 1:max_iter_power
        q_old = q_proj;
        
        % Power step
        q_proj = G_mat_rect * q_proj;
        
        % Project: only keep diagonal positions (k,k) → idx = (k-1)*r + k
        q_proj_temp = zeros(r*r, 1);
        for k = 1:r
            idx = (k-1)*r + k;
            q_proj_temp(idx) = q_proj(idx);
        end
        q_proj = q_proj_temp;
        
        % Normalize
        q_norm = norm(q_proj);
        if q_norm > 1e-10
            q_proj = q_proj / q_norm;
        else
            warning('q_proj became zero at iteration %d', iter);
            break;
        end
        
        % Check convergence
        conv_err = min(norm(q_proj - q_old), norm(q_proj + q_old));
        if conv_err < tol_power
            fprintf('    Converged at iteration %d (conv_err = %.6e)\n', iter, conv_err);
            break;
        end
        
        if mod(iter, 10) == 0
            fprintf('    Iter %d: convergence = %.6e\n', iter, conv_err);
        end
    end
    
    % Rayleigh quotient
    lambda_proj = q_proj' * G_mat_rect * q_proj;
    fprintf('  Estimated eigenvalue: %.6f\n', lambda_proj);
    
    % Reconstruct
    v_proj = U_kron * q_proj;
    X_method3 = reshape(v_proj * sqrt(abs(lambda_proj)), [d1, d2]);
    X_method3 = X_method3 / norm(X_method3, 'fro');
    
    time_method3 = toc;
    
    error_method3 = min(norm(X_method3 - X_true, 'fro'), ...
                        norm(X_method3 + X_true, 'fro'));
    rank_method3 = rank(X_method3, 1e-10);
    fprintf('  X_method3: %dx%d (diagonal-projected)\n', d1, d2);
    fprintf('  Rank of X_method3: %d\n', rank_method3);
    fprintf('  Reconstruction error: %.6e\n', error_method3);
    fprintf('  Time: %.4f seconds\n', time_method3);
    
    % Analyze diagonal support
    fprintf('\n  Diagonal support analysis:\n');
    fprintf('    Non-zero positions in q_proj: ');
    num_nonzero = 0;
    for k = 1:r
        idx = (k-1)*r + k;
        if abs(q_proj(idx)) > 1e-10
            num_nonzero = num_nonzero + 1;
        end
    end
    fprintf('%d / %d diagonal positions\n', num_nonzero, r);
    fprintf('    Diagonal values: [');
    for k = 1:r
        idx = (k-1)*r + k;
        fprintf('%.3f ', q_proj(idx));
    end
    fprintf(']\n');
    
    % Compare all three methods
    fprintf('\n=== Three Method Comparison ===\n');
    diff_12 = min(norm(X_method1 - X_method2, 'fro'), ...
                  norm(X_method1 + X_method2, 'fro'));
    diff_13 = min(norm(X_method1 - X_method3, 'fro'), ...
                  norm(X_method1 + X_method3, 'fro'));
    diff_23 = min(norm(X_method2 - X_method3, 'fro'), ...
                  norm(X_method2 + X_method3, 'fro'));
    
    fprintf('Result agreement:\n');
    fprintf('  ||X_method1 - X_method2||_F = %.6e\n', diff_12);
    fprintf('  ||X_method1 - X_method3||_F = %.6e\n', diff_13);
    fprintf('  ||X_method2 - X_method3||_F = %.6e\n', diff_23);
    
    max_diff = max([diff_12, diff_13, diff_23]);
    if max_diff < 1e-6
        fprintf('  ✓ All three methods produce identical results\n');
    elseif max_diff < 1e-3
        fprintf('  ✓ All methods produce similar results\n');
    else
        fprintf('  ~ Methods show some differences (max diff: %.2e)\n', max_diff);
    end
    
    fprintf('\nComputational efficiency:\n');
    fprintf('  Method 1 time: %.4f seconds (eigendecompose %dx%d matrix)\n', ...
            time_method1, d1*d2, d1*d2);
    fprintf('  Method 2 time: %.4f seconds (eigendecompose %dx%d matrix)\n', ...
            time_method2, r*r, r*r);
    fprintf('  Method 3 time: %.4f seconds (projected power on %dx%d matrix)\n', ...
            time_method3, r*r, r*r);
    fprintf('  Speedup (M2 vs M1): %.2fx\n', time_method1 / time_method2);
    fprintf('  Speedup (M3 vs M1): %.2fx\n', time_method1 / time_method3);
    
    if time_method2 < time_method1 && time_method2 < time_method3
        fprintf('  ✓ Method 2 is fastest (DEFAULT)\n');
    elseif time_method3 < time_method1 && time_method3 < time_method2
        fprintf('  ✓ Method 3 (projected power) is fastest\n');
    else
        fprintf('  ~ Method 1 is fastest (for very small problems)\n');
    end
    
    fprintf('\nReconstruction quality:\n');
    fprintf('  Method 1 error: %.6e\n', error_method1);
    fprintf('  Method 2 error: %.6e (DEFAULT)\n', error_method2);
    fprintf('  Method 3 error: %.6e (diagonal-projected)\n', error_method3);
    
    % Find best method
    [min_error, best_idx] = min([error_method1, error_method2, error_method3]);
    if min_error < 0.01
        fprintf('  ✓ Excellent reconstruction (< 1%%) by Method %d\n', best_idx);
    elseif min_error < 0.1
        fprintf('  ✓ Good reconstruction (< 10%%) by Method %d\n', best_idx);
    else
        fprintf('  ~ Moderate reconstruction (best: Method %d)\n', best_idx);
    end
    
    % Rank analysis
    fprintf('\n⚠️  RANK ANALYSIS ⚠️\n');
    fprintf('  Ground truth X_true: rank = %d\n', r);
    fprintf('  Method 1 X_method1:  rank = %d\n', rank_method1);
    fprintf('  Method 2 X_method2:  rank = %d\n', rank_method2);
    fprintf('  Method 3 X_method3:  rank = %d\n', rank_method3);
    
    % Singular value analysis for Method 2 (DEFAULT)
    fprintf('\n  Singular value analysis (Method 2 - DEFAULT):\n');
    [U_svd, S_svd, V_svd] = svd(X_method2);
    s_vals = diag(S_svd);
    fprintf('    Top %d singular values:\n', min(r+3, length(s_vals)));
    for i = 1:min(r+3, length(s_vals))
        if i <= r
            fprintf('      σ_%d = %.6e  ← expected signal\n', i, s_vals(i));
        else
            fprintf('      σ_%d = %.6e  ← numerical noise\n', i, s_vals(i));
        end
    end
    
    % Check if effectively rank-r
    if r < length(s_vals)
        ratio = s_vals(r+1) / s_vals(1);
        fprintf('    Rank-r gap: σ_%d / σ_1 = %.2e\n', r+1, ratio);
        if ratio < 1e-6
            fprintf('    ✓ Effectively rank-%d (gap > 10^6)\n', r);
        elseif ratio < 1e-3
            fprintf('    ✓ Approximately rank-%d (gap > 10^3)\n', r);
        else
            fprintf('    ⚠️  NOT clearly rank-%d (small gap)\n', r);
        end
    end
    
    % Interpretation
    fprintf('\n  💡 INTERPRETATION:\n');
    fprintf('     Methods extract v_max ≈ vec(X_true)\n');
    fprintf('     X_recovered = reshape(v_max, [d1,d2]) should preserve rank!\n');
    fprintf('     If rank ≈ r: ✓ Algorithm working correctly\n');
    fprintf('     If rank = 1: ⚠️  Signal may be rank-1, or noise dominated\n');
    
    fprintf('\nRecommendation:\n');
    if error_method3 < error_method2 * 0.9
        fprintf('  ✓✓ Use Method 3 (extract_matrix_from_tucker_3)\n');
        fprintf('      Better accuracy (%.2f%% improvement) by exploiting diagonal structure\n', ...
                (1 - error_method3/error_method2) * 100);
    elseif abs(error_method3 - error_method2) < 0.01 * error_method2
        fprintf('  ✓ Method 2 and 3 are equivalent in accuracy\n');
        fprintf('    Use Method 2 (slightly faster, DEFAULT)\n');
    else
        fprintf('  ✓ Use Method 2 (extract_matrix_from_tucker_2) - DEFAULT\n');
        fprintf('    For r=%d << min(d1,d2)=%d\n', r, min(d1, d2));
    end
end

%% Computational Efficiency Comparison
fprintf('\n=== Step 8: Computational Efficiency ===\n');
fprintf('Computation times:\n');
fprintf('  Tensor mode products: %.4f seconds\n', time_tensor);
fprintf('  Matrix formula:       %.4f seconds\n', time_formula);
fprintf('  Speedup: %.2fx\n', time_tensor / time_formula);

if time_formula < time_tensor
    fprintf('  ✓ Matrix formula is faster\n');
else
    fprintf('  Note: Tensor mode products competitive for this size\n');
end

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('Test case: %s symmetry\n', test_case);
fprintf('  Matrix dimensions: %dx%d (d1×d2)\n', d1, d2);
fprintf('  Factor relationships:\n');
fprintf('    ||U1 - U3||_F = %.2e\n', diff_U13);
fprintf('    ||U2 - U4||_F = %.2e\n', diff_U24);
fprintf('\nFormula verification:\n');
fprintf('  G_mat formula: %.2e (relative diff)\n', rel_diff_G);
if rel_diff_G < 1e-6
    fprintf('  ✓ PASS\n');
else
    fprintf('  ✗ FAIL\n');
end

fprintf('\nSymmetry check:\n');
fprintf('  H_mat: %.2e (relative asymmetry)\n', H_mat_symm_error);
fprintf('  G_mat: %.2e (relative asymmetry)\n', G_mat_symm_error);
if d1 ~= d2
    fprintf('  Note: X is %dx%d (non-square) → asymmetry expected\n', d1, d2);
end

if strcmp(test_case, 'partial')
    fprintf('\nMatrix extraction (partial symmetry):\n');
    fprintf('  Method 1 error: %.2e\n', error_method1);
    fprintf('  Method 2 error: %.2e (DEFAULT)\n', error_method2);
    fprintf('  Method 3 error: %.2e (diagonal-projected)\n', error_method3);
    fprintf('  Method 1-2 agreement: %.2e\n', diff_12);
    fprintf('  Method 2-3 agreement: %.2e\n', diff_23);
    
    % Check if Method 3 improves over Method 2
    if error_method3 < error_method2 * 0.9
        fprintf('  ✓✓ Method 3 improves accuracy by %.1f%%\n', ...
                (1 - error_method3/error_method2) * 100);
    end
    
    if rel_diff_G < 1e-6 && max([diff_12, diff_23]) < 1e-3 && ...
       min([error_method1, error_method2, error_method3]) < 0.1
        fprintf('\n✓ All tests PASSED for partial symmetry case\n');
    else
        fprintf('\n~ Some tests show moderate performance\n');
    end
else
    if rel_diff_G < 1e-6
        fprintf('\n✓ Formula test PASSED for general non-symmetric case\n');
    else
        fprintf('\n✗ Formula test FAILED\n');
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

