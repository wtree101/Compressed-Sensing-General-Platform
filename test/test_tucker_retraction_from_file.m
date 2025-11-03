% test_tucker_retraction_from_file.m
% Test Tucker tensor retraction using saved tensors from debug mode
%
% This script loads T_tucker and Grad_F saved during the initialization
% and tests whether retraction matches HOSVD approach

clear; clc;
% addpath('../utilities_tensor');
% addpath('../utilities');
% addpath('../Initialization_groundtruth');

fprintf('=== Testing Tucker Retraction from Saved Data ===\n\n');

%% Load saved debug data
data_file = '/Users/wutong/Documents/MATLAB/GeneralPlatform/debug_tucker_tensors.mat';
if ~exist(data_file, 'file')
    error('Debug data file not found. Run test_tucker_lift_init.m with debug=true first.');
end

fprintf('Loading saved tensors from: %s\n', data_file);
load(data_file, 'debug_data');

T = debug_data.T_tucker;
Grad_F = debug_data.Grad_F;
eta = debug_data.eta;
tucker_ranks = debug_data.tucker_rank * ones(1, 4);
dims = debug_data.dims;

fprintf('  Loaded successfully!\n');
fprintf('  Tensor dimensions: [%s]\n', num2str(dims));
fprintf('  Tucker ranks: [%s]\n', num2str(tucker_ranks));
fprintf('  Step size eta: %.4f\n', eta);
fprintf('  T.G norm: %.6f\n', norm(T.G(:)));
fprintf('  Grad_F.G norm: %.6f\n', norm(Grad_F.G(:)));
fprintf('\n');

%% Check tensor properties
fprintf('=== Tensor Properties ===\n');
fprintf('T_tucker:\n');
fprintf('  Core G: size [%s], norm=%.6f\n', num2str(size(T.G)), norm(T.G(:)));
fprintf('  U{1}: size [%dx%d], orthogonal=%s\n', size(T.U{1}), ...
        mat2str(norm(T.U{1}'*T.U{1} - eye(tucker_ranks(1)), 'fro') < 1e-10));
fprintf('  Symmetric: %s\n', mat2str(T.is_symmetric));

fprintf('\nGrad_F:\n');
fprintf('  Core G: size [%s], norm=%.6f\n', num2str(size(Grad_F.G)), norm(Grad_F.G(:)));
if ~isempty(Grad_F.Up{1})
    fprintf('  Up{1}: size [%dx%d], norm=%.6f\n', size(Grad_F.Up{1}), norm(Grad_F.Up{1}, 'fro'));
    % Check orthogonality to U
    orth_check = norm(T.U{1}' * Grad_F.Up{1}, 'fro');
    fprintf('  Up orthogonal to U: %s (||U''*Up||=%.2e)\n', mat2str(orth_check < 1e-10), orth_check);
else
    fprintf('  Up{1}: empty\n');
end
fprintf('\n');

%% Method 1: Use retraction function
fprintf('=== Method 1: Retraction ===\n');
tic;
T_retracted = T.retraction(Grad_F, eta);
time_retraction = toc;

fprintf('  Retraction completed in %.4f seconds\n', time_retraction);
fprintf('  Retracted core norm: %.6f\n', norm(T_retracted.G(:)));
fprintf('\n');

%% Method 2: Direct HOSVD on full tensor
fprintf('=== Method 2: Full Tensor HOSVD ===\n');
tic;

% Step 1: Form full tensors
fprintf('  Forming full tensors...\n');
T_full = T.full();

% Construct full gradient manually
fprintf('  Constructing full gradient tensor...\n');

% First term: Grad_F.G ×₁ U₁ ×₂ U₂ ×₃ ... ×_N U_N
if isscalar(Grad_F.G)
    % Special case: rank-1 tensor (scalar core)
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
        % Build Kronecker product with Up_i in position i
        term_factors = cell(1, length(dims));
        for j = 1:length(dims)
            if j == i
                term_factors{j} = Grad_F.Up{j};
            else
                term_factors{j} = T.U{j};
            end
        end
        
        % Form Kronecker product
        term = term_factors{1};
        for j = 2:length(dims)
            term = kron(term, term_factors{j});
        end
        term = reshape(term, dims) * T.G;
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

fprintf('    T_full norm: %.6f\n', norm(T_full(:)));
fprintf('    Grad_full norm: %.6f\n', norm(Grad_full(:)));

% Step 2: Gradient descent step on full tensor
T_updated_full = T_full - eta * Grad_full;
fprintf('  Updated tensor norm: %.6f\n', norm(T_updated_full(:)));

% Step 3: Apply HOSVD to truncate back to Tucker ranks
fprintf('  Applying HOSVD...\n');
T_hosvd_full = HOSVD(T_updated_full, tucker_ranks);

time_hosvd = toc;
fprintf('  HOSVD completed in %.4f seconds\n', time_hosvd);
fprintf('  HOSVD result norm: %.6f\n', norm(T_hosvd_full(:)));
fprintf('\n');

%% Method 3: Form retracted tensor in full format for comparison
fprintf('=== Method 3: Converting Results to Full Tensors ===\n');
T_retracted_full = T_retracted.full();
fprintf('  Retracted full tensor norm: %.6f\n', norm(T_retracted_full(:)));
fprintf('\n');

%% Compare results
fprintf('=== Comparison Results ===\n');

% Compute relative difference
diff = T_retracted_full - T_hosvd_full;
rel_error = norm(diff(:)) / (norm(T_hosvd_full(:)) + 1e-15);

fprintf('Relative error: %.6e\n', rel_error);
fprintf('Absolute difference norm: %.6e\n', norm(diff(:)));
fprintf('HOSVD result norm: %.6e\n', norm(T_hosvd_full(:)));
fprintf('Retraction result norm: %.6e\n', norm(T_retracted_full(:)));
fprintf('\n');

% Element-wise statistics
fprintf('Element-wise comparison:\n');
fprintf('  Max absolute difference: %.6e\n', max(abs(diff(:))));
fprintf('  Mean absolute difference: %.6e\n', mean(abs(diff(:))));
fprintf('  Std of difference: %.6e\n', std(diff(:)));
fprintf('\n');

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
fprintf('\n');

%% Debug: Compare Tucker components
fprintf('=== Detailed Component Comparison ===\n');

% Compare cores
if isscalar(T_retracted.G)
    fprintf('Retracted core: scalar = %.6f\n', T_retracted.G);
else
    fprintf('Retracted core G norm: %.6f\n', norm(T_retracted.G(:)));
end

% Extract Tucker decomposition from HOSVD result
T_hosvd_tucker = hosvd_decompose(T_hosvd_full, tucker_ranks);
if isscalar(T_hosvd_tucker.G)
    fprintf('HOSVD core: scalar = %.6f\n', T_hosvd_tucker.G);
else
    fprintf('HOSVD core G norm: %.6f\n', norm(T_hosvd_tucker.G(:)));
end

% Compare core tensors
if ~isscalar(T_retracted.G) && ~isscalar(T_hosvd_tucker.G)
    core_diff = norm(T_retracted.G(:) - T_hosvd_tucker.G(:)) / norm(T_hosvd_tucker.G(:));
    fprintf('Core relative difference: %.6e\n', core_diff);
end

% Compare factor matrices (with sign ambiguity)
fprintf('\nFactor matrix differences:\n');
for i = 1:4
    % Try both signs
    diff_pos = norm(T_retracted.U{i} - T_hosvd_tucker.U{i}, 'fro');
    diff_neg = norm(T_retracted.U{i} + T_hosvd_tucker.U{i}, 'fro');
    diff_min = min(diff_pos, diff_neg);
    fprintf('  U{%d}: %.6e\n', i, diff_min);
end
fprintf('\n');

%% Visualization
figure('Position', [100, 100, 1400, 500]);

% Plot 1: Slice comparison
subplot(1,4,1);
slice_idx = 1;
slice_retraction = squeeze(T_retracted_full(slice_idx, :, :, slice_idx));
imagesc(slice_retraction);
colorbar;
title('Retraction: T(1,:,:,1)');
xlabel('Mode-3'); ylabel('Mode-2');

subplot(1,4,2);
slice_hosvd = squeeze(T_hosvd_full(slice_idx, :, :, slice_idx));
imagesc(slice_hosvd);
colorbar;
title('HOSVD: T(1,:,:,1)');
xlabel('Mode-3'); ylabel('Mode-2');

% Plot 2: Difference
subplot(1,4,3);
slice_diff = slice_retraction - slice_hosvd;
imagesc(slice_diff);
colorbar;
title(sprintf('Difference (max: %.2e)', max(abs(slice_diff(:)))));
xlabel('Mode-3'); ylabel('Mode-2');

% Plot 3: Relative error histogram
subplot(1,4,4);
rel_diff = abs(diff(:)) ./ (abs(T_hosvd_full(:)) + 1e-15);
histogram(log10(rel_diff + 1e-16), 50);
xlabel('log10(Relative Error)');
ylabel('Frequency');
title('Relative Error Distribution');
grid on;

sgtitle(sprintf('Tucker Retraction Test from File: %s (rel\\_error = %.2e)', test_status, rel_error));

%% Performance comparison
fprintf('=== Performance ===\n');
fprintf('Retraction time: %.4f seconds\n', time_retraction);
fprintf('HOSVD time: %.4f seconds\n', time_hosvd);
fprintf('Speedup: %.2fx\n', time_hosvd / time_retraction);
fprintf('\n');

%% Summary
fprintf('=== Summary ===\n');
fprintf('Test status: %s\n', test_status);
fprintf('Relative error: %.6e (tolerance: %.0e)\n', rel_error, tolerance);
if rel_error < tolerance
    fprintf('The retraction IS equivalent to HOSVD truncation.\n');
else
    fprintf('The retraction is NOT equivalent to HOSVD truncation.\n');
    fprintf('Possible reasons for discrepancy:\n');
    fprintf('  1. Gradient construction for rank-1 case may have bugs\n');
    fprintf('  2. Up components may not be properly orthogonal to U\n');
    fprintf('  3. HOSVD sign/ordering ambiguity\n');
    fprintf('  4. Numerical precision issues\n');
end

%% Helper functions
function T_tucker = hosvd_decompose(T_full, tucker_ranks)
    % Perform HOSVD decomposition
    dims = size(T_full);
    N = length(dims);
    
    if isscalar(tucker_ranks)
        tucker_ranks = tucker_ranks * ones(1, N);
    end
    
    U_cells = cell(1, N);
    
    for n = 1:N
        T_unfold = mode_n_unfold(T_full, n, dims);
        [U, ~, ~] = svd(T_unfold, 'econ');
        U_cells{n} = U(:, 1:tucker_ranks(n));
    end
    
    G = T_full;
    for n = 1:N
        G = tensor_mode_product(G, U_cells{n}', n);
    end
    
    T_tucker = struct();
    T_tucker.G = G;
    T_tucker.U = U_cells;
end

function T_mat = mode_n_unfold(T, n, dims)
    N = length(dims);
    perm = [n, 1:n-1, n+1:N];
    T_perm = permute(T, perm);
    T_mat = reshape(T_perm, dims(n), []);
end

function T_out = tensor_mode_product(T, M, mode)
    sz = size(T);
    k = size(M, 1);
    
    order = 1:max(ndims(T), mode);
    order([1, mode]) = [mode, 1];
    T_perm = permute(T, order);
    
    T_mat = reshape(T_perm, sz(mode), []);
    T_out_mat = M * T_mat;
    
    sz_out = sz;
    sz_out(mode) = k;
    sz_out_perm = sz_out(order);
    T_out_perm = reshape(T_out_mat, sz_out_perm);
    
    T_out = ipermute(T_out_perm, order);
end
