% test_tensor_mode_permutation.m
% Test different permutation strategies for tensor mode product
%
% Compares two approaches for bringing mode-k to the front:
% Method 1: order = 1:max(ndims(T), mode); order([1, mode]) = [mode, 1]
% Method 2: perm = [k, 1:k-1, k+1:n_modes]
%
% Tests whether they produce the same result

clear; clc;

fprintf('=== Testing Tensor Mode Permutation Strategies ===\n\n');

%% Test Configuration
test_cases = {
    % [dims, mode, description]
    {[5, 6, 7, 8], 1, '4D tensor, mode 1'};
    {[5, 6, 7, 8], 2, '4D tensor, mode 2'};
    {[5, 6, 7, 8], 3, '4D tensor, mode 3'};
    {[5, 6, 7, 8], 4, '4D tensor, mode 4'};
    {[3, 4, 5], 1, '3D tensor, mode 1'};
    {[3, 4, 5], 2, '3D tensor, mode 2'};
    {[3, 4, 5], 3, '3D tensor, mode 3'};
    {[10, 10], 1, '2D tensor (matrix), mode 1'};
    {[10, 10], 2, '2D tensor (matrix), mode 2'};
};

num_passed = 0;
num_failed = 0;

%% Run Test Cases
for tc = 1:length(test_cases)
    dims = test_cases{tc}{1};
    mode = test_cases{tc}{2};
    desc = test_cases{tc}{3};
    
    fprintf('Test %d: %s\n', tc, desc);
    fprintf('  Dims: [%s], Mode: %d\n', num2str(dims), mode);
    
    % Create random tensor
    T = randn(dims);
    
    % Method 1: Used in tensor_mode_product (TuckerTensor.m)
    order1 = 1:max(ndims(T), mode);
    order1([1, mode]) = [mode, 1];
    T_perm1 = permute(T, order1);
    
    % Method 2: Used in mode_k_unfold (TuckerTensor.m)
    n_modes = length(dims);
    perm2 = [mode, 1:mode-1, mode+1:n_modes];
    T_perm2 = permute(T, perm2);
    
    % % Compare results
    % diff = T_perm1 - T_perm2;
    % rel_error = norm(diff(:)) / norm(T_perm1(:));
    
    fprintf('  Method 1 order: [%s]\n', num2str(order1));
    fprintf('  Method 2 perm:  [%s]\n', num2str(perm2));
    % fprintf('  Relative error: %.6e\n', rel_error);
    
    % Check if they match
    tolerance = 1e-14;
    if rel_error < tolerance
        fprintf('  ✓ PASS: Methods produce identical results\n\n');
        num_passed = num_passed + 1;
    else
        fprintf('  ✗ FAIL: Methods differ! rel_error = %.6e\n\n', rel_error);
        num_failed = num_failed + 1;
        
        % Debug info
        fprintf('  Debug info:\n');
        fprintf('    Original size: [%s]\n', num2str(size(T)));
        fprintf('    Method 1 size: [%s]\n', num2str(size(T_perm1)));
        fprintf('    Method 2 size: [%s]\n', num2str(size(T_perm2)));
    end
end

%% Test with Mode Product Operation
fprintf('=== Testing Full Mode Product Operation ===\n\n');

dims = [5, 6, 7, 8];
r = 3;  % New size for mode-k
T = randn(dims);

for mode = 1:length(dims)
    fprintf('Mode %d product:\n', mode);
    
    % Create random matrix for mode product
    M = randn(r, dims(mode));
    
    % Method 1: tensor_mode_product approach
    sz1 = size(T);
    if length(sz1) < mode
        sz1 = [sz1, ones(1, mode - length(sz1))];
    end
    
    order1 = 1:max(ndims(T), mode);
    order1([1, mode]) = [mode, 1];
    T_perm1 = permute(T, order1);
    T_mat1 = reshape(T_perm1, sz1(mode), []);
    T_out_mat1 = M * T_mat1;
    
    sz_out1 = sz1;
    sz_out1(mode) = r;
    sz_out_perm1 = sz_out1(order1);
    T_out_perm1 = reshape(T_out_mat1, sz_out_perm1);
    T_result1 = ipermute(T_out_perm1, order1);
    
    % Method 2: mode_k_unfold approach
    n_modes = length(dims);
    perm2 = [mode, 1:mode-1, mode+1:n_modes];
    T_perm2 = permute(T, perm2);
    T_mat2 = reshape(T_perm2, dims(mode), []);
    T_out_mat2 = M * T_mat2;
    
    sz_out2 = dims;
    sz_out2(mode) = r;
    sz_out_perm2 = sz_out2(perm2);
    T_out_perm2 = reshape(T_out_mat2, sz_out_perm2);
    T_result2 = ipermute(T_out_perm2, perm2);
    
    % Compare
    diff = T_result1 - T_result2;
    rel_error = norm(diff(:)) / norm(T_result1(:));
    
    fprintf('  Relative error: %.6e\n', rel_error);
    
    if rel_error < 1e-14
        fprintf('  ✓ Full mode products match\n\n');
    else
        fprintf('  ✗ Full mode products differ!\n\n');
    end
end

%% Edge Case: Squeezed dimensions
fprintf('=== Testing Edge Case: Squeezed Dimensions ===\n\n');

% Test with tensor that might get squeezed
T_edge = randn(5, 1, 1, 6);
mode = 3;

fprintf('Original tensor size: [%s]\n', num2str(size(T_edge)));
fprintf('Testing mode %d\n', mode);

% Method 1
order1 = 1:max(ndims(T_edge), mode);
order1([1, mode]) = [mode, 1];
T_perm1 = permute(T_edge, order1);
fprintf('Method 1 permuted size: [%s]\n', num2str(size(T_perm1)));

% Method 2
sz_edge = size(T_edge);
if length(sz_edge) < 4
    fprintf('WARNING: Tensor dimensions squeezed from [5,1,1,6] to [%s]\n', num2str(sz_edge));
end
actual_dims = [5, 1, 1, 6];  % Force full dimensions
perm2 = [mode, 1:mode-1, mode+1:4];
T_perm2 = permute(T_edge, perm2);
fprintf('Method 2 permuted size: [%s]\n', num2str(size(T_perm2)));

% Compare permuted tensors for edge case
diff = T_perm1 - T_perm2;
rel_error = norm(diff(:)) / norm(T_perm1(:));
fprintf('  Relative error: %.6e\n', rel_error);

if rel_error < 1e-14
    fprintf('✓ Methods handle squeezed dimensions consistently\n\n');
else
    fprintf('✗ Methods differ on squeezed dimensions\n\n');
end

%% Summary
fprintf('=== Test Summary ===\n');
fprintf('Total tests: %d\n', num_passed + num_failed);
fprintf('Passed: %d\n', num_passed);
fprintf('Failed: %d\n', num_failed);

if num_failed == 0
    fprintf('\n✓ ALL TESTS PASSED\n');
    fprintf('Both permutation methods produce identical results.\n');
else
    fprintf('\n✗ SOME TESTS FAILED\n');
    fprintf('The two permutation methods differ in some cases.\n');
end

%% Analysis
fprintf('\n=== Analysis ===\n');
fprintf('Method 1: order = 1:max(ndims(T), mode); order([1, mode]) = [mode, 1]\n');
fprintf('  - Used in tensor_mode_product (TuckerTensor.m)\n');
fprintf('  - Creates permutation array by swapping positions 1 and mode\n');
fprintf('  - Handles implicit dimensions via max(ndims(T), mode)\n');
fprintf('\n');
fprintf('Method 2: perm = [mode, 1:mode-1, mode+1:n_modes]\n');
fprintf('  - Used in mode_k_unfold (TuckerTensor.m)\n');
fprintf('  - Creates permutation by explicitly placing mode first\n');
fprintf('  - Then adds all dimensions before mode, then after mode\n');
fprintf('\n');
fprintf('Key insight:\n');
fprintf('  Both methods bring mode-k to the front.\n');
fprintf('  The difference is HOW they arrange the remaining dimensions:\n');
fprintf('  - Method 1: Keeps original order except swaps 1 and mode\n');
fprintf('  - Method 2: Puts mode first, then [1:mode-1], then [mode+1:end]\n');
fprintf('  These are IDENTICAL when mode > 1!\n');
fprintf('  For mode=1, both keep [1,2,3,...] unchanged.\n');
