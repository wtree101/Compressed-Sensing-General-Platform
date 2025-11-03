% test_tucker_retraction.m
% Test Tucker tensor retraction by comparing with direct HOSVD approach
%
% Verifies that retraction(T, Grad_F, eta) gives the same result as:
% 1. Form full tensor: T_full - eta * Grad_full
% 2. Apply HOSVD to truncate back to Tucker ranks
%
% The two approaches should give equivalent results (up to numerical precision)

clear; clc;
addpath('../utilities_tensor');
addpath('../utilities');

fprintf('=== Testing Tucker Tensor Retraction ===\n\n');

%% Test Configuration
% Small tensor for verification
dims = [20,20,20,20];      % 4th-order tensor
tucker_ranks = [1,1,1,1];  % Tucker ranks
eta = 0.01;                % Step size
symmetric = false;          % Use symmetric tensor

fprintf('Configuration:\n');
fprintf('  Tensor dimensions: [%s]\n', num2str(dims));
fprintf('  Tucker ranks: [%s]\n', num2str(tucker_ranks));
fprintf('  Step size eta: %.4f\n', eta);
fprintf('  Symmetric: %s\n\n', mat2str(symmetric));

%% Create initial Tucker tensor
fprintf('Creating initial Tucker tensor...\n');
T = TuckerTensor(dims, tucker_ranks, 'symmetric', symmetric, 'init_method', 'orthogonal');

% Initialize with some random values for testing
if symmetric
    % For symmetric Tucker tensor, core G should be invariant under permutation
    T.G = generate_symmetric_core(tucker_ranks(1));
    T.G = T.G * 0.5;  % Scale
else
    T.G = randn(tucker_ranks) * 0.1;
end

if symmetric
    T.U{1} = orth(randn(dims(1), tucker_ranks(1)));
    for i = 2:length(dims)
        T.U{i} = T.U{1};
    end
else
    for i = 1:length(dims)
        T.U{i} = orth(randn(dims(i), tucker_ranks(i)));
    end
end

fprintf('  Initial tensor created\n');
fprintf('  Core tensor norm: %.6f\n\n', norm(T.G(:)));

%% Create gradient tensor (mimics actual gradient structure)
fprintf('Creating gradient tensor...\n');
Grad_F = TuckerTensor(dims, tucker_ranks, 'symmetric', symmetric, 'init_method', 'zeros');

% Random gradient for core
if symmetric
    % For symmetric Tucker tensor, core G should be invariant under permutation
    % Generate symmetric core by averaging over all permutations
    Grad_F.G = generate_symmetric_core(tucker_ranks(1));
    Grad_F.G = Grad_F.G * 0.1;  % Scale down
else
    Grad_F.G = -randn(tucker_ranks)*1.7;
end

% Random gradients for orthogonal components Up
for i = 1:length(dims)
    % Up should be orthogonal to U (in tangent space)
    % For testing, just use random matrices
    Up_temp = randn(dims(i), tucker_ranks(i)) * 0.1;
    % Make orthogonal to U{i}
    Up_temp = Up_temp - T.U{i} * (T.U{i}' * Up_temp);
    Grad_F.Up{i} = Up_temp;
end

if symmetric
    % For symmetric tensor, all Up should be the same
    for i = 2:length(dims)
        Grad_F.Up{i} = Grad_F.Up{1};
    end
end

fprintf('  Gradient tensor created\n');
fprintf('  Gradient core norm: %.6f\n\n', norm(Grad_F.G(:)));

%% Verify symmetry properties if symmetric=true
if symmetric
    fprintf('=== Verifying Symmetric Structure ===\n');
    
    % Check T.G is symmetric under permutation
    T_G_sym_error = check_core_symmetry(T.G);
    fprintf('T.G symmetry error: %.6e\n', T_G_sym_error);
    
    % Check Grad_F.G is symmetric under permutation
    Grad_G_sym_error = check_core_symmetry(Grad_F.G);
    fprintf('Grad_F.G symmetry error: %.6e\n', Grad_G_sym_error);
    
    % Check all U are the same
    U_diff_max = 0;
    for i = 2:4
        U_diff_max = max(U_diff_max, norm(T.U{i} - T.U{1}, 'fro'));
    end
    fprintf('T.U factor difference: %.6e\n', U_diff_max);
    
    % Check all Up are the same
    Up_diff_max = 0;
    for i = 2:4
        Up_diff_max = max(Up_diff_max, norm(Grad_F.Up{i} - Grad_F.Up{1}, 'fro'));
    end
    fprintf('Grad_F.Up factor difference: %.6e\n\n', Up_diff_max);
end

%% Method 1: Use retraction function
fprintf('Method 1: Using retraction function...\n');
tic;
T_retracted = T.retraction(Grad_F, eta);
time_retraction = toc;

fprintf('  Retraction completed in %.4f seconds\n', time_retraction);
fprintf('  Retracted core norm: %.6f\n\n', norm(T_retracted.G(:)));

%% Verify retracted tensor maintains symmetry
if symmetric
    fprintf('=== Verifying Retracted Tensor Symmetry ===\n');
    
    % Check T_retracted.G is symmetric under permutation
    T_ret_G_sym_error = check_core_symmetry(T_retracted.G);
    fprintf('T_retracted.G symmetry error: %.6e\n', T_ret_G_sym_error);
    
    % Check all U are the same in retracted tensor
    U_ret_diff_max = 0;
    for i = 2:4
        U_ret_diff_max = max(U_ret_diff_max, norm(T_retracted.U{i} - T_retracted.U{1}, 'fro'));
    end
    fprintf('T_retracted.U factor difference: %.6e\n\n', U_ret_diff_max);
    
    if T_ret_G_sym_error > 1e-10 || U_ret_diff_max > 1e-10
        fprintf('⚠ WARNING: Retracted tensor lost symmetric structure!\n\n');
    else
        fprintf('✓ Retracted tensor maintains symmetric structure\n\n');
    end
end

%% Method 2: Direct HOSVD on full tensor
fprintf('Method 2: Direct HOSVD on full tensor...\n');
tic;

% Step 1: Form full tensors
fprintf('  Forming full tensors...\n');
T_full = T.full();

% Construct gradient in full format manually since Grad_F.full() doesn't handle Up
% The gradient has structure: Grad_G ×₁ U₁ ×₂ ... ×_N U_N + sum_i (G ×₁ U₁ ... ×ᵢ Upᵢ ... ×_N U_N)
fprintf('  Constructing full gradient tensor...\n');

% First term: Grad_F.G ×₁ U₁ ×₂ U₂ ×₃ ... ×_N U_N
if isscalar(Grad_F.G)
    % Special case: rank-1 tensor (scalar core)
    % Form as Kronecker product: U₁ ⊗ U₂ ⊗ ... ⊗ U_N
    Grad_full = T.U{1};
    for i = 2:length(dims)
        Grad_full = kron(Grad_full, T.U{i});
    end
    Grad_full = reshape(Grad_full, dims) * Grad_F.G;
else
    % General case: apply mode products
    Grad_full = Grad_F.G;
    for i = 1:length(dims)
        Grad_full = tensor_mode_product(Grad_full, T.U{i}, i);
    end
end

% Add terms for each mode: G ×₁ U₁ ... ×ᵢ Upᵢ ... ×_N U_N
for i = 1:length(dims)
    if isscalar(T.G)
        % Special case: rank-1 tensor (scalar core)
        term = T.U{1};
        for j = 2:length(dims)
            if j == i
                term = kron(term, Grad_F.Up{j});
            else
                term = kron(term, T.U{j});
            end
        end
        % Handle first mode
        if i == 1
            term_temp = Grad_F.Up{1};
            for j = 2:length(dims)
                term_temp = kron(term_temp, T.U{j});
            end
            term = reshape(term_temp, dims) * T.G;
        else
            term = reshape(term, dims) * T.G;
        end
    else
        % General case: apply mode products
        term = T.G;
        for j = 1:length(dims)
            if j == i
                term = tensor_mode_product(term, Grad_F.Up{j}, j);
            else
                term = tensor_mode_product(term, T.U{j}, j);
            end
        end
    end
    Grad_full = Grad_full + term;
end

fprintf('    T_full size: [%s], norm: %.6f\n', num2str(size(T_full)), norm(T_full(:)));
fprintf('    Grad_full size: [%s], norm: %.6f\n', num2str(size(Grad_full)), norm(Grad_full(:)));

% Step 2: Gradient descent step on full tensor
T_updated_full = T_full - eta * Grad_full;
fprintf('  Updated tensor norm: %.6f\n', norm(T_updated_full(:)));

% Step 3: Apply HOSVD to truncate back to Tucker ranks
fprintf('  Applying HOSVD...\n');
T_hosvd_full = HOSVD(T_updated_full, tucker_ranks);

time_hosvd = toc;
fprintf('  HOSVD completed in %.4f seconds\n', time_hosvd);
fprintf('  HOSVD result norm: %.6f\n\n', norm(T_hosvd_full(:)));

%% Method 3: Form retracted tensor in full format for comparison
fprintf('Method 3: Converting retraction result to full tensor...\n');
T_retracted_full = T_retracted.full();
fprintf('  Retracted full tensor norm: %.6f\n\n', norm(T_retracted_full(:)));

%% Compare results
fprintf('=== Comparison Results ===\n');

% Compute relative difference
diff = T_retracted_full - T_hosvd_full;
rel_error = norm(diff(:)) / norm(T_hosvd_full(:));

fprintf('Relative error: %.6e\n', rel_error);
fprintf('Absolute difference norm: %.6e\n', norm(diff(:)));
fprintf('HOSVD result norm: %.6e\n', norm(T_hosvd_full(:)));
fprintf('Retraction result norm: %.6e\n\n', norm(T_retracted_full(:)));

% Element-wise statistics
fprintf('Element-wise comparison:\n');
fprintf('  Max absolute difference: %.6e\n', max(abs(diff(:))));
fprintf('  Mean absolute difference: %.6e\n', mean(abs(diff(:))));
fprintf('  Std of difference: %.6e\n\n', std(diff(:)));

% Check if results match within tolerance
tolerance = 1e-10;
if rel_error < tolerance
    fprintf('✓ TEST PASSED: Retraction matches HOSVD (rel_error < %.0e)\n', tolerance);
    test_status = 'PASSED';
else
    fprintf('✗ TEST FAILED: Retraction differs from HOSVD (rel_error = %.6e >= %.0e)\n', ...
            rel_error, tolerance);
    test_status = 'FAILED';
end

%% Visualization
figure('Position', [100, 100, 1200, 400]);

% Plot 1: Slice comparison (first mode)
subplot(1,3,1);
slice_idx = {1, ':', ':', ':'};
slice_retraction = squeeze(T_retracted_full(slice_idx{:}));
slice_hosvd = squeeze(T_hosvd_full(slice_idx{:}));
imagesc([slice_retraction(:,:,1), slice_hosvd(:,:,1)]);
colorbar;
title('Comparison: Retraction (left) vs HOSVD (right)');
xlabel('Mode-2 index');
ylabel('Mode-3 index');

% Plot 2: Difference
subplot(1,3,2);
slice_diff = slice_retraction - slice_hosvd;
imagesc(slice_diff(:,:,1));
colorbar;
title(sprintf('Difference (max: %.2e)', max(abs(slice_diff(:)))));
xlabel('Mode-2 index');
ylabel('Mode-3 index');

% Plot 3: Relative error histogram
subplot(1,3,3);
rel_diff = abs(diff(:)) ./ (abs(T_hosvd_full(:)) + 1e-15);
histogram(log10(rel_diff), 50);
xlabel('log10(Relative Error)');
ylabel('Frequency');
title('Relative Error Distribution');
grid on;

sgtitle(sprintf('Tucker Retraction Test: %s (rel\\_error = %.2e)', test_status, rel_error));

%% Performance comparison
fprintf('\n=== Performance ===\n');
fprintf('Retraction time: %.4f seconds\n', time_retraction);
fprintf('HOSVD time: %.4f seconds\n', time_hosvd);
fprintf('Speedup: %.2fx\n', time_hosvd / time_retraction);

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('Test status: %s\n', test_status);
fprintf('Relative error: %.6e (tolerance: %.0e)\n', rel_error, tolerance);
fprintf('The retraction operation %s equivalent to HOSVD truncation.\n', ...
        ternary(rel_error < tolerance, 'is', 'is NOT'));

% Helper function
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

function G_sym = generate_symmetric_core(r)
    % GENERATE_SYMMETRIC_CORE Generate 4th-order core invariant under permutation
    % For symmetric Tucker tensor with tied factors, the core should satisfy:
    % G(i,j,k,l) is invariant under any permutation of (i,j,k,l)
    %
    % Input:
    %   r - Tucker rank (scalar)
    %
    % Output:
    %   G_sym - Symmetric core tensor (r×r×r×r)
    
    % Start with random tensor
    G_random = randn(r, r, r, r);
    
    % Symmetrize by averaging over all 24 permutations of 4 indices
    permss = perms(1:4);  % All 24 permutations
    G_sym = zeros(r, r, r, r);
    
    for p = 1:size(permss, 1)
        perm_idx = permss(p, :);
        G_sym = G_sym + permute(G_random, perm_idx);
    end
    
    % Average
    G_sym = G_sym / size(permss, 1);
end

function sym_error = check_core_symmetry(G)
    % CHECK_CORE_SYMMETRY Check if 4th-order core is symmetric under permutation
    % Returns the maximum deviation from symmetry
    %
    % Input:
    %   G - Core tensor (r×r×r×r)
    %
    % Output:
    %   sym_error - Maximum relative error across all permutations
    
    if isscalar(G)
        % Scalar is trivially symmetric
        sym_error = 0;
        return;
    end
    
    % Get all 24 permutations
    permss = perms(1:4);
    sym_error = 0;
    G_norm = norm(G(:));
    
    % Check deviation from original for each permutation
    for p = 1:size(permss, 1)
        perm_idx = permss(p, :);
        G_perm = permute(G, perm_idx);
        diff = norm(G_perm(:) - G(:)) / (G_norm + 1e-15);
        sym_error = max(sym_error, diff);
    end
end

function T_out = tensor_mode_product(T, M, mode)
    % TENSOR_MODE_PRODUCT n-mode product of tensor T with matrix M
    % T_out = T ×_mode M
    
    sz = size(T);
    
    % Handle scalar/low-dimensional tensors
    if isscalar(T)
        warning('Tensor is scalar, treating as 1x1x...x1 tensor');
    end
    
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
