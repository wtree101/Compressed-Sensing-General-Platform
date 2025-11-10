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
m = 800;             % Number of measurements
use_debug = true;    % Enable debug mode for detailed output

rng(42);  % For reproducibility

fprintf('Test Configuration:\n');
fprintf('  Dimension d: %d\n', d);
fprintf('  Tucker rank r: %d\n', r);
fprintf('  Measurements m: %d\n', m);
fprintf('  Debug mode: %s\n\n', mat2str(use_debug));

%% Generate Ground Truth (symmetric rank-r matrix)
fprintf('Generating ground truth...\n');
U_true = randn(d, r);
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
    %A_cells{i} = (Ai + Ai') / 2;  % Symmetrize
    A_cells{i} = Ai;
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
fprintf('\n--- Method 2: Tucker Decomposition (U * C_root * U^T) ---\n');

% Since T = X ⊗ X and X = U * C_root * U^T (for symmetric case)
% The core tensor G should have structure related to C_root ⊗ C_root
%
% Strategy:
% 1. Contract G with U^T on modes 3 and 4 to eliminate those modes
% 2. This gives us a (r × r × d × d) tensor
% 3. Further process to extract C_root

U = U_cell{1};  % All factors are the same due to symmetry

fprintf('Extracting C_root from core tensor G...\n');

% Method 2a: Direct extraction assuming G = C_root ⊗ C_root
% For symmetric Tucker: T = G ×₁ U ×₂ U ×₃ U ×₄ U
% If X = U * C_root * U^T, then ideally G should relate to C_root ⊗ C_root

if isscalar(G_init)
    % Special case: scalar core (rank-1)
    C_root = sqrt(abs(G_init)) * eye(1);
    fprintf('  Scalar core: G = %.6f\n', G_init);
    fprintf('  C_root: %.6f (scalar)\n', C_root);
else
    % General case: extract C_root from G
    % One approach: matricize G and extract square root structure
    
    % Matricize G to (r² × r²)
    G_mat = reshape(G_init, [r*r, r*r]);
    fprintf('  G matricized: %dx%d matrix\n', r*r, r*r);
    fprintf('  G_mat norm: %.6f\n', norm(G_mat, 'fro'));
    
    % Symmetrize
    G_mat = (G_mat + G_mat') / 2;
    
    % Extract leading eigenvector/eigenvalue
    [V_g, D_g] = eig(G_mat);
    [~, idx_g] = max(abs(diag(D_g)));
    v_g = V_g(:, idx_g);
    lambda_g = D_g(idx_g, idx_g);
    
    fprintf('  Leading eigenvalue of G_mat: %.6f\n', lambda_g);
    
    % Reshape to get C_root (r × r)
    % The eigenvector represents vec(C_root ⊗ C_root)
    % So we need to extract C_root from this
    
    % Simple approach: reshape and take matrix square root
    C_temp = reshape(v_g * sqrt(abs(lambda_g)), [r, r]);
    
    % Since we want C_root such that C_temp ≈ C_root ⊗ C_root (as vector)
    % For rank-1 case, C_root is approximately the square root
    % For general case, we can use eigendecomposition
    
    C_temp = (C_temp + C_temp') / 2;  % Symmetrize
    
    % Extract C_root via eigendecomposition
    [V_c, D_c] = eig(C_temp);
    [~, idx_c] = max(abs(diag(D_c)));
    
    % For rank-1 Tucker: C_root ≈ sqrt(leading_eigenvalue) * v * v'
    % For higher rank: take leading eigenpair
    if r == 1
        C_root = sqrt(abs(D_c(1,1)));
        fprintf('  C_root (rank-1): %.6f\n', C_root);
    else
        % Take top eigenvalues and form low-rank approximation
        [~, sort_idx] = sort(abs(diag(D_c)), 'descend');
        D_c_sorted = D_c(sort_idx, sort_idx);
        V_c_sorted = V_c(:, sort_idx);
        
        % Form C_root from leading eigenpairs
        % Apply fourth root since G ≈ (C_root ⊗ C_root)
        D_c_root = diag(nthroot(abs(diag(D_c_sorted)), 4) .* sign(diag(D_c_sorted)));
        C_root = V_c_sorted * D_c_root * V_c_sorted';
        
        fprintf('  C_root extracted: %dx%d matrix, norm=%.6f\n', r, r, norm(C_root, 'fro'));
    end
end

% Form X_method2 = U * C_root * U^T
if isscalar(C_root)
    X_method2 = U * C_root * U';
else
    X_method2 = U * C_root * U';
end

% Normalize and symmetrize
X_method2 = X_method2 / norm(X_method2, 'fro');
X_method2 = (X_method2 + X_method2') / 2;

fprintf('  X_method2: %dx%d matrix, norm=%.6f\n', d, d, norm(X_method2, 'fro'));
fprintf('  X_method2 rank: %d\n', rank(X_method2, 1e-6));

% Compute reconstruction error for Method 2
[recon_error_method2, X_method2] = rectify_sign_ambiguity(X_method2, Xstar);
fprintf('  Reconstruction error: %.6e\n', recon_error_method2);

if recon_error_method2 < 0.1
    fprintf('  ✓ Good reconstruction\n');
elseif recon_error_method2 < 0.5
    fprintf('  ~ Moderate reconstruction\n');
else
    fprintf('  ✗ Poor reconstruction\n');
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

