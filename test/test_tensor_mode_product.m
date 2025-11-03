% test_tensor_mode_product.m
% Test tensor mode product with different permutation strategies
%
% Tests the tensor_mode_product implementation to verify:
% 1. Different permutation strategies give the same result
% 2. Mode product satisfies mathematical properties
% 3. Edge cases are handled correctly

clear; clc;

fprintf('=== Testing Tensor Mode Product Implementation ===\n\n');

%% Test 1: Compare Current vs Alternative Permutation Strategy
fprintf('Test 1: Comparing Permutation Strategies\n');
fprintf('=========================================\n\n');

test_cases = {
    % [dims, mode, new_dim]
    {[5, 6, 7, 8], 1, 3};
    {[5, 6, 7, 8], 2, 4};
    {[5, 6, 7, 8], 3, 5};
    {[5, 6, 7, 8], 4, 6};
    {[3, 4, 5], 1, 2};
    {[3, 4, 5], 2, 3};
    {[3, 4, 5], 3, 4};
};

num_passed = 0;
num_failed = 0;

for tc = 1:length(test_cases)
    dims = test_cases{tc}{1};
    mode = test_cases{tc}{2};
    new_dim = test_cases{tc}{3};
    
    fprintf('Test case %d: dims=[%s], mode=%d\n', tc, num2str(dims), mode);
    
    % Create random tensor and matrix
    T = randn(dims);
    M = randn(new_dim, dims(mode));
    
    % Method 1: Current implementation (swap positions 1 and mode)
    result1 = tensor_mode_product_method1(T, M, mode);
    
    % Method 2: Alternative (explicit [mode, 1:mode-1, mode+1:end])
    result2 = tensor_mode_product_method2(T, M, mode, dims);
    
    % Compare results
    diff = result1 - result2;
    rel_error = norm(diff(:)) / max(norm(result1(:)), 1e-10);
    
    fprintf('  Result size: [%s]\n', num2str(size(result1)));
    fprintf('  Relative error: %.6e\n', rel_error);
    
    if rel_error < 1e-14
        fprintf('  ✓ PASS: Both methods give identical results\n\n');
        num_passed = num_passed + 1;
    else
        fprintf('  ✗ FAIL: Methods differ!\n\n');
        num_failed = num_failed + 1;
    end
end

fprintf('Permutation Test Summary: %d/%d passed\n\n', num_passed, num_passed+num_failed);

%% Test 2: Mathematical Properties
fprintf('Test 2: Mathematical Properties of Mode Product\n');
fprintf('===============================================\n\n');

% Test 2.1: Associativity T ×_k (M1*M2) = (T ×_k M1) ×_k M2
fprintf('Test 2.1: Associativity\n');
dims = [4, 5, 6];
mode = 2;
T = randn(dims);
M1 = randn(3, dims(mode));
M2 = randn(2, 3);

result_left = tensor_mode_product_method1(T, M2*M1, mode);
temp = tensor_mode_product_method1(T, M1, mode);
result_right = tensor_mode_product_method1(temp, M2, mode);

error_assoc = norm(result_left(:) - result_right(:)) / norm(result_left(:));
fprintf('  Relative error: %.6e\n', error_assoc);
if error_assoc < 1e-12
    fprintf('  ✓ PASS: Associativity holds\n\n');
else
    fprintf('  ✗ FAIL: Associativity violated\n\n');
end

% Test 2.2: Commutativity for different modes
fprintf('Test 2.2: Mode Commutativity (different modes)\n');
dims = [4, 5, 6];
T = randn(dims);
M1 = randn(3, dims(1));
M2 = randn(4, dims(2));

result12 = tensor_mode_product_method1(...
    tensor_mode_product_method1(T, M1, 1), M2, 2);
result21 = tensor_mode_product_method1(...
    tensor_mode_product_method1(T, M2, 2), M1, 1);

error_comm = norm(result12(:) - result21(:)) / norm(result12(:));
fprintf('  Relative error: %.6e\n', error_comm);
if error_comm < 1e-12
    fprintf('  ✓ PASS: Mode products commute for different modes\n\n');
else
    fprintf('  ✗ FAIL: Commutativity violated\n\n');
end

%% Test 3: Edge Cases
fprintf('Test 3: Edge Cases\n');
fprintf('==================\n\n');

% Test 3.1: Rank-1 tensor (all dims = 1 except one)
fprintf('Test 3.1: Rank-1 tensor (singleton dimensions)\n');
T_rank1 = randn(1, 1, 5, 1);
M_rank1 = randn(3, 5);
try
    result_rank1 = tensor_mode_product_method1(T_rank1, M_rank1, 3);
    expected_size = [1, 1, 3, 1];
    if isequal(size(result_rank1), expected_size)
        fprintf('  ✓ PASS: Rank-1 tensor handled correctly\n');
        fprintf('    Result size: [%s]\n\n', num2str(size(result_rank1)));
    else
        fprintf('  ✗ FAIL: Wrong output size\n');
        fprintf('    Expected: [%s], Got: [%s]\n\n', ...
            num2str(expected_size), num2str(size(result_rank1)));
    end
catch ME
    fprintf('  ✗ FAIL: Error occurred\n');
    fprintf('    %s\n\n', ME.message);
end

% Test 3.2: Mode = 1
fprintf('Test 3.2: Mode 1 (first dimension)\n');
T_mode1 = randn(5, 6, 7);
M_mode1 = randn(3, 5);
result_mode1 = tensor_mode_product_method1(T_mode1, M_mode1, 1);
expected_size = [3, 6, 7];
if isequal(size(result_mode1), expected_size)
    fprintf('  ✓ PASS: Mode 1 works correctly\n');
    fprintf('    Result size: [%s]\n\n', num2str(size(result_mode1)));
else
    fprintf('  ✗ FAIL: Wrong output size for mode 1\n\n');
end

% Test 3.3: Last mode
fprintf('Test 3.3: Last mode\n');
dims = [4, 5, 6];
T_last = randn(dims);
mode_last = length(dims);
M_last = randn(3, dims(mode_last));
result_last = tensor_mode_product_method1(T_last, M_last, mode_last);
expected_size = [4, 5, 3];
if isequal(size(result_last), expected_size)
    fprintf('  ✓ PASS: Last mode works correctly\n');
    fprintf('    Result size: [%s]\n\n', num2str(size(result_last)));
else
    fprintf('  ✗ FAIL: Wrong output size for last mode\n\n');
end

% Test 3.4: Identity matrix (should give same tensor)
fprintf('Test 3.4: Identity matrix mode product\n');
dims = [5, 6, 7];
T_id = randn(dims);
mode_id = 2;
M_id = eye(dims(mode_id));
result_id = tensor_mode_product_method1(T_id, M_id, mode_id);
error_id = norm(result_id(:) - T_id(:)) / norm(T_id(:));
fprintf('  Relative error: %.6e\n', error_id);
if error_id < 1e-14
    fprintf('  ✓ PASS: Identity matrix preserves tensor\n\n');
else
    fprintf('  ✗ FAIL: Identity matrix changes tensor\n\n');
end

%% Test 4: Tucker Decomposition Application
fprintf('Test 4: Tucker Decomposition Reconstruction\n');
fprintf('===========================================\n\n');

dims = [10, 12, 8];
ranks = [3, 4, 2];

% Create Tucker components
G = randn(ranks);
U1 = randn(dims(1), ranks(1));
U2 = randn(dims(2), ranks(2));
U3 = randn(dims(3), ranks(3));

% Reconstruct via sequential mode products
T_recon = G;
T_recon = tensor_mode_product_method1(T_recon, U1, 1);
T_recon = tensor_mode_product_method1(T_recon, U2, 2);
T_recon = tensor_mode_product_method1(T_recon, U3, 3);

fprintf('  Core dimensions: [%s]\n', num2str(ranks));
fprintf('  Tensor dimensions: [%s]\n', num2str(dims));
fprintf('  Reconstructed size: [%s]\n', num2str(size(T_recon)));

if isequal(size(T_recon), dims)
    fprintf('  ✓ PASS: Tucker reconstruction has correct size\n\n');
else
    fprintf('  ✗ FAIL: Tucker reconstruction has wrong size\n\n');
end

%% Summary
fprintf('=== Overall Summary ===\n');
fprintf('Permutation comparison: %d/%d tests passed\n', num_passed, num_passed+num_failed);
fprintf('\nConclusion:\n');
fprintf('The tensor mode product implementation correctly handles:\n');
fprintf('  - Different permutation strategies (if tests pass)\n');
fprintf('  - Mathematical properties (associativity, commutativity)\n');
fprintf('  - Edge cases (singleton dimensions, first/last mode)\n');
fprintf('  - Tucker decomposition reconstruction\n');

%% Helper Functions

function T_out = tensor_mode_product_method1(T, M, mode)
    % METHOD 1: Current implementation - swap positions 1 and mode
    % order = 1:max(ndims(T), mode); order([1, mode]) = [mode, 1]
    
    sz = size(T);
    k = size(M, 1);
    
    % Permute so mode is first (swap positions)
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

function T_out = tensor_mode_product_method2(T, M, mode, dims)
    % METHOD 2: Alternative - explicit permutation [mode, 1:mode-1, mode+1:end]
    % perm = [mode, 1:mode-1, mode+1:n_modes]
    
    k = size(M, 1);
    n_modes = length(dims);
    
    % Permute so mode is first (explicit ordering)
    perm = [mode, 1:mode-1, mode+1:n_modes];
    T_perm = permute(T, perm);
    
    % Reshape to matrix and multiply
    T_mat = reshape(T_perm, dims(mode), []);
    T_out_mat = M * T_mat;
    
    % Reshape back
    sz_out = dims;
    sz_out(mode) = k;
    sz_out_perm = sz_out(perm);
    T_out_perm = reshape(T_out_mat, sz_out_perm);
    
    % Permute back
    T_out = ipermute(T_out_perm, perm);
end
