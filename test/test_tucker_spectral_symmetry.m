%% Test TuckerTensor.initialize_spectral for Symmetry
% This test verifies that initialize_spectral produces symmetric factor matrices
% i.e., U{1} = U{2} = U{3} = U{4}
%
% Test Date: 2025-11-03

clear; clc;

fprintf('=== Test TuckerTensor.initialize_spectral Symmetry ===\n\n');

%% Test Parameters
d = 20;              % Dimension
r = 2;               % Tucker rank
m = 300;             % Number of measurements
use_debug = true;    % Enable debug mode for detailed output

%rng(42);  % For reproducibility

fprintf('Test Configuration:\n');
fprintf('  Dimension d: %d\n', d);
fprintf('  Tucker rank r: %d\n', r);
fprintf('  Measurements m: %d\n', m);
fprintf('  Debug mode: %s\n\n', mat2str(use_debug));

%% Generate Ground Truth (symmetric rank-r matrix)
fprintf('Generating ground truth...\n');
U_true = randn(d, r);
V_true = randn(d, r);
Xstar = U_true * U_true';  % Symmetric matrix
Xstar = Xstar / norm(Xstar, 'fro');
fprintf('  Ground truth: %dx%d symmetric matrix, rank=%d\n\n', d, d, r);

%% Create Measurement Operator
fprintf('Creating measurement operator...\n');
A = randn(m, d*d);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d, d]);

% Extract measurement matrices (A_cells)
A_cells = cell(m, 1);
for i = 1:m
    Ai = reshape(A(i, :), [d, d]);
    A_cells{i} = (Ai + Ai') / 2;  % Symmetrize
    %A_cells{i} = Ai;
end

spectral_operator = struct();
spectral_operator.A_cells = A_cells;

fprintf('  Created %d measurement matrices\n\n', m);

%% Generate Measurements (phase retrieval)
fprintf('Generating measurements (phase retrieval)...\n');
y = abs(operator.A(Xstar)) / sqrt(m);
fprintf('  Measurements: %d values\n', length(y));
fprintf('  Measurement norm: %.6f\n\n', norm(y));

%% Convert to Tensor Measurements
% For spectral init: y_spectral_i = y_i^2 * sqrt(m)
y_spectral = y.^2 * sqrt(m);
fprintf('Tensor measurements: norm = %.6f\n\n', norm(y_spectral));

%% Create TuckerTensor Object and Call initialize_spectral
fprintf('=== Running initialize_spectral ===\n\n');

dims = [d, d, d, d];
T_tucker = TuckerTensor(dims, r, 'symmetric', false, 'init_method', 'zeros', 'debug', use_debug);

tic;
[U_cell, G_init] = T_tucker.initialize_spectral(spectral_operator, y_spectral, m);
elapsed_time = toc;

fprintf('\n=== initialize_spectral Complete ===\n');
fprintf('Execution time: %.4f seconds\n\n', elapsed_time);

%% Check Symmetry: U{1} = U{2} = U{3} = U{4}
fprintf('=== Checking Factor Matrix Symmetry ===\n\n');

% Compare all factor matrices
is_symmetric = true;
tolerance = 1e-10;  % Numerical tolerance

fprintf('Comparing factor matrices (tolerance = %.2e):\n', tolerance);

% Check U{1} vs U{2}
diff_12 = norm(U_cell{1} - U_cell{2}, 'fro');
fprintf('  ||U{1} - U{2}||_F = %.6e\n', diff_12);
if diff_12 > tolerance
    is_symmetric = false;
end

% Check U{1} vs U{3}
diff_13 = norm(U_cell{1} - U_cell{3}, 'fro');
fprintf('  ||U{1} - U{3}||_F = %.6e\n', diff_13);
if diff_13 > tolerance
    is_symmetric = false;
end

% Check U{1} vs U{4}
diff_14 = norm(U_cell{1} - U_cell{4}, 'fro');
fprintf('  ||U{1} - U{4}||_F = %.6e\n', diff_14);
if diff_14 > tolerance
    is_symmetric = false;
end

% Check U{2} vs U{3}
diff_23 = norm(U_cell{2} - U_cell{3}, 'fro');
fprintf('  ||U{2} - U{3}||_F = %.6e\n', diff_23);
if diff_23 > tolerance
    is_symmetric = false;
end

% Check U{2} vs U{4}
diff_24 = norm(U_cell{2} - U_cell{4}, 'fro');
fprintf('  ||U{2} - U{4}||_F = %.6e\n', diff_24);
if diff_24 > tolerance
    is_symmetric = false;
end

% Check U{3} vs U{4}
diff_34 = norm(U_cell{3} - U_cell{4}, 'fro');
fprintf('  ||U{3} - U{4}||_F = %.6e\n', diff_34);
if diff_34 > tolerance
    is_symmetric = false;
end

max_diff = max([diff_12, diff_13, diff_14, diff_23, diff_24, diff_34]);
fprintf('\n  Maximum difference: %.6e\n', max_diff);

%% Test Result
fprintf('\n=== Test Result ===\n');
if is_symmetric
    fprintf('✓ PASS: All factor matrices are identical (symmetric)\n');
    fprintf('  U{1} = U{2} = U{3} = U{4}\n');
else
    fprintf('✗ FAIL: Factor matrices are NOT identical\n');
    fprintf('  Maximum difference %.6e exceeds tolerance %.6e\n', max_diff, tolerance);
end

%% Additional Checks
fprintf('\n=== Additional Checks ===\n');

% Check orthogonality of U{1}
fprintf('\nOrthogonality check for U{1}:\n');
orthogonality_error = norm(U_cell{1}' * U_cell{1} - eye(r), 'fro');
fprintf('  ||U{1}^T U{1} - I||_F = %.6e\n', orthogonality_error);
if orthogonality_error < 1e-6
    fprintf('  ✓ U{1} is orthogonal\n');
else
    fprintf('  ✗ U{1} is NOT orthogonal (may be intentional)\n');
end

% Check core tensor
fprintf('\nCore tensor G:\n');
fprintf('  Size: %s\n', mat2str(size(G_init)));
fprintf('  Norm: %.6f\n', norm(G_init(:)));
fprintf('  Min value: %.6e\n', min(G_init(:)));
fprintf('  Max value: %.6e\n', max(G_init(:)));

% Check if core is diagonal (for symmetric case, might be diagonal)
if ndims(G_init) == 4 && all(size(G_init) == [r, r, r, r])
    % Extract diagonal elements (if r is small)
    if r <= 5
        fprintf('  Core tensor structure:\n');
        for i = 1:r
            fprintf('    G(%d,%d,%d,%d) = %.6f\n', i, i, i, i, G_init(i,i,i,i));
        end
    end
end

%% Reconstruct Matrix and Compare with Ground Truth
fprintf('\n=== Reconstruction Check ===\n');

% Create TuckerTensor with initialized factors
T_result = TuckerTensor(dims, r, 'symmetric', false, 'init_method', 'zeros');
T_result.U = U_cell;
T_result.G = G_init;

fprintf('Testing two extraction methods:\n');
fprintf('  Method 1: Eigendecomposition of matricized tensor\n');
fprintf('  Method 2: Tucker decomposition (U * C_root * U^T)\n\n');

%% Method 1: Extract matrix from tensor (matricization + eigenvector)
fprintf('--- Method 1: Matricization + Eigendecomposition ---\n');

% For 4th-order tensor T = X ⊗ X, matricize and extract leading eigenvector
U1 = U_cell{1};
U2 = U_cell{2};
U3 = U_cell{3};
U4 = U_cell{4};

% Matricized tensor: T_mat = (U₁ ⊗ U₂) * G_mat * (U₃ ⊗ U₄)'
if isscalar(G_init)
    U_left = kron(U1, U2);
    U_right = kron(U3, U4);
    T_mat = G_init * (U_left * U_right');
else
    G_mat = reshape(permute(G_init, [1,2,3,4]), [r*r, r*r]);
    U_left = kron(U1, U2);
    U_right = kron(U3, U4);
    T_mat = U_left * G_mat * U_right';
end

% Symmetrize
T_mat = (T_mat + T_mat') / 2;

% Extract leading eigenvector
[V, D] = eig(T_mat);
[~, idx] = max(abs(diag(D)));
v_lead = V(:, idx);
X_reconstructed = reshape(v_lead, [d, d]);
X_reconstructed = X_reconstructed / norm(X_reconstructed, 'fro');
X_reconstructed = (X_reconstructed + X_reconstructed') / 2;

% Compute reconstruction error
[recon_error_method1, X_reconstructed] = rectify_sign_ambiguity(X_reconstructed, Xstar);
fprintf('  Reconstruction error: %.6e\n', recon_error_method1);
fprintf('  Reconstruction rank: %d\n', rank(X_reconstructed, 1e-6));

if recon_error_method1 < 0.1
    fprintf('  ✓ Good reconstruction\n');
elseif recon_error_method1 < 0.5
    fprintf('  ~ Moderate reconstruction\n');
else
    fprintf('  ✗ Poor reconstruction\n');
end

%% Method 2: Extract matrix using Tucker decomposition structure
fprintf('\n--- Method 2: Diagonal Core Extraction (Tucker structure) ---\n');

% Step 1: Extract diagonal part of core tensor G
% Form matrix G_2(i,j) = G(i,i,j,j) (ignore non-diagonal parts)
fprintf('Step 1: Extracting diagonal part of core tensor\n');
fprintf('  Forming G_2(i,j) = G(i,i,j,j)...\n');

r_val = size(G_init, 1);  % Tucker rank
G_2 = zeros(r_val, r_val);

for i = 1:r_val
    for j = 1:r_val
        G_2(i, j) = G_init(i, i, j, j);
    end
end

fprintf('  G_2 size: %dx%d\n', size(G_2, 1), size(G_2, 2));
fprintf('  G_2 norm: %.6f\n', norm(G_2, 'fro'));

% Step 2: Check symmetry of G_2
fprintf('\nStep 2: Checking symmetry of G_2\n');
G_2_symmetric = (G_2 + G_2') / 2;
G_2_asymmetric = norm(G_2 - G_2_symmetric, 'fro');
fprintf('  ||G_2 - G_2^T||_F = %.6e\n', G_2_asymmetric);

if G_2_asymmetric < 1e-10
    fprintf('  ✓ G_2 is symmetric\n');
else
    fprintf('  ✗ G_2 is NOT symmetric (asymmetry: %.6e)\n', G_2_asymmetric);
    fprintf('  Symmetrizing G_2 for computation...\n');
    G_2 = G_2_symmetric;
end

% Step 3: Compute spectral decomposition of G_2
fprintf('\nStep 3: Computing spectral decomposition of G_2\n');
[V_G2, D_G2] = eig(G_2);
[eigenvalues_sorted, idx_sort] = sort(diag(D_G2), 'descend');

fprintf('  Eigenvalues of G_2 (sorted):\n');
for k = 1:min(r_val, 3)
    fprintf('    λ{%d} = %.6e\n', k, eigenvalues_sorted(k));
end

% Step 4: Extract eigenvector corresponding to largest eigenvalue
fprintf('\nStep 4: Extracting eigenvector of largest eigenvalue\n');
lambda_max = eigenvalues_sorted(1);
v_G2_max = V_G2(:, idx_sort(1));

fprintf('  Largest eigenvalue: λ_max = %.6e\n', lambda_max);
fprintf('  Eigenvector norm: %.6f\n', norm(v_G2_max));

% Step 5: Reconstruct matrix from Tucker factors and eigenvector
fprintf('\nStep 5: Reconstructing matrix from Tucker factors\n');
% X = U₁ * v * U₁^T (since U₁ = U₂ = U₃ = U₄ for symmetric case)
U_factor = U_cell{1};
X_method2 = U_factor * diag(v_G2_max) * U_factor';

% Ensure symmetry
X_method2 = (X_method2 + X_method2') / 2;

fprintf('  Reconstructed matrix size: %dx%d\n', size(X_method2, 1), size(X_method2, 2));
fprintf('  Reconstructed matrix norm: %.6f\n', norm(X_method2, 'fro'));

% Normalize
X_method2_norm = norm(X_method2, 'fro');
if X_method2_norm > 0
    X_method2 = X_method2 / X_method2_norm;
end

% Compute reconstruction error for Method 2
[recon_error_method2, X_method2] = rectify_sign_ambiguity(X_method2, Xstar);
fprintf('  Reconstruction error: %.6e\n', recon_error_method2);
fprintf('  Reconstruction rank: %d\n', rank(X_method2, 1e-6));

if recon_error_method2 < 0.1
    fprintf('  ✓ Good reconstruction\n');
elseif recon_error_method2 < 0.5
    fprintf('  ~ Moderate reconstruction\n');
else
    fprintf('  ✗ Poor reconstruction\n');
end

% Additional diagnostic: print G_2 matrix
fprintf('\nDiagnostic: G_2 matrix (first %d×%d):\n', min(r_val, 5), min(r_val, 5));
for i = 1:min(r_val, 5)
    fprintf('  ');
    for j = 1:min(r_val, 5)
        fprintf('%8.4f ', G_2(i, j));
    end
    fprintf('\n');
end

%% Compare the two methods
fprintf('\n--- Comparison of Extraction Methods ---\n');
fprintf('Method 1 (Matricization) error: %.6e\n', recon_error_method1);
fprintf('Method 2 (Tucker U*C*U^T) error: %.6e\n', recon_error_method2);

% Compare the extracted matrices directly
diff_methods = norm(X_reconstructed - X_method2, 'fro');
fprintf('Difference between methods: ||X1 - X2||_F = %.6e\n', diff_methods);

if diff_methods < 1e-6
    fprintf('✓ Both methods produce identical results\n');
elseif diff_methods < 1e-3
    fprintf('~ Methods produce similar results\n');
else
    fprintf('✗ Methods produce different results\n');
end

% Use the better method for final reconstruction error
recon_error = min(recon_error_method1, recon_error_method2);
fprintf('\nBest reconstruction error: %.6e (Method %d)\n', ...
        recon_error, 1 + (recon_error_method2 < recon_error_method1));

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('Symmetry test: %s\n', mat2str(is_symmetric));
fprintf('  Max difference between U{i}: %.6e\n', max_diff);
fprintf('\nExecution time: %.4f seconds\n', elapsed_time);
fprintf('\nReconstruction results:\n');
fprintf('  Method 1 (Matricization):   %.6e\n', recon_error_method1);
fprintf('  Method 2 (Tucker U*C*U^T):  %.6e\n', recon_error_method2);
fprintf('  Best reconstruction error:  %.6e\n', recon_error);
fprintf('  Methods difference:         %.6e\n', diff_methods);

fprintf('\nTest Results:\n');
if is_symmetric
    fprintf('  ✓ Symmetry: PASS (U{1}=U{2}=U{3}=U{4})\n');
else
    fprintf('  ✗ Symmetry: FAIL\n');
end

if recon_error < 0.1
    fprintf('  ✓ Reconstruction: EXCELLENT (error < 0.1)\n');
elseif recon_error < 0.5
    fprintf('  ✓ Reconstruction: GOOD (error < 0.5)\n');
else
    fprintf('  ✗ Reconstruction: POOR (error >= 0.5)\n');
end

if diff_methods < 1e-3
    fprintf('  ✓ Method consistency: PASS (both methods agree)\n');
else
    fprintf('  ~ Method consistency: WARNING (methods differ by %.2e)\n', diff_methods);
end

if is_symmetric && recon_error < 0.5
    fprintf('\n✓ All tests PASSED\n');
else
    fprintf('\n✗ Some tests FAILED\n');
end

fprintf('\n=== Test Complete ===\n');

