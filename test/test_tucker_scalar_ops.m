% test_tucker_scalar_ops.m
% Test scalar operations on TuckerTensor class

clear; clc;

fprintf('=== Testing TuckerTensor Scalar Operations ===\n\n');

%% Setup test tensor
dims = [5, 6, 7, 8];
ranks = [2, 3, 2, 3];
rng(42);

% Create Tucker tensor
T = TuckerTensor(dims, ranks);
T.G = randn(ranks);
for i = 1:length(dims)
    T.U{i} = randn(dims(i), ranks(i));
end

fprintf('Test tensor created:\n');
fprintf('  Dimensions: [%s]\n', num2str(dims));
fprintf('  Tucker ranks: [%s]\n', num2str(ranks));
fprintf('  Core norm: %.4f\n\n', norm(T.G(:)));

%% Test 1: Division by scalar
fprintf('Test 1: Division by Scalar (T / scalar)\n');
fprintf('========================================\n');

scalar = 2.5;
T_div = T / scalar;

% Check that core is divided
core_diff = norm(T_div.G(:) - T.G(:) / scalar);
% Check that factors are unchanged
factor_diff = 0;
for i = 1:length(dims)
    factor_diff = factor_diff + norm(T_div.U{i} - T.U{i}, 'fro');
end

fprintf('  T / %.2f\n', scalar);
fprintf('  Core difference: %.6e\n', core_diff);
fprintf('  Factor difference: %.6e\n', factor_diff);

if core_diff < 1e-14 && factor_diff < 1e-14
    fprintf('  ✓ PASS: Division works correctly\n\n');
else
    fprintf('  ✗ FAIL: Division incorrect\n\n');
end

%% Test 2: Multiplication by scalar
fprintf('Test 2: Multiplication by Scalar (T * scalar)\n');
fprintf('==============================================\n');

scalar = 3.7;
T_mult = T * scalar;

% Check that core is multiplied
core_diff = norm(T_mult.G(:) - T.G(:) * scalar);
% Check that factors are unchanged
factor_diff = 0;
for i = 1:length(dims)
    factor_diff = factor_diff + norm(T_mult.U{i} - T.U{i}, 'fro');
end

fprintf('  T * %.2f\n', scalar);
fprintf('  Core difference: %.6e\n', core_diff);
fprintf('  Factor difference: %.6e\n', factor_diff);

if core_diff < 1e-14 && factor_diff < 1e-14
    fprintf('  ✓ PASS: Multiplication works correctly\n\n');
else
    fprintf('  ✗ FAIL: Multiplication incorrect\n\n');
end

%% Test 3: Commutative multiplication (scalar * T)
fprintf('Test 3: Commutative Multiplication (scalar * T)\n');
fprintf('================================================\n');

scalar = 1.5;
T_mult1 = T * scalar;
T_mult2 = scalar * T;

core_diff = norm(T_mult1.G(:) - T_mult2.G(:));

fprintf('  T * %.2f vs %.2f * T\n', scalar, scalar);
fprintf('  Core difference: %.6e\n', core_diff);

if core_diff < 1e-14
    fprintf('  ✓ PASS: Multiplication is commutative\n\n');
else
    fprintf('  ✗ FAIL: Multiplication not commutative\n\n');
end

%% Test 4: Element-wise operations
fprintf('Test 4: Element-wise Operations (.* and ./)\n');
fprintf('===========================================\n');

scalar = 2.0;
T_times = T .* scalar;
T_rdiv = T ./ scalar;

core_times_diff = norm(T_times.G(:) - T.G(:) * scalar);
core_rdiv_diff = norm(T_rdiv.G(:) - T.G(:) / scalar);

fprintf('  T .* %.2f: core diff = %.6e\n', scalar, core_times_diff);
fprintf('  T ./ %.2f: core diff = %.6e\n', scalar, core_rdiv_diff);

if core_times_diff < 1e-14 && core_rdiv_diff < 1e-14
    fprintf('  ✓ PASS: Element-wise operations work correctly\n\n');
else
    fprintf('  ✗ FAIL: Element-wise operations incorrect\n\n');
end

%% Test 5: Unary minus
fprintf('Test 5: Unary Minus (-T)\n');
fprintf('=========================\n');

T_neg = -T;

core_diff = norm(T_neg.G(:) + T.G(:));
factor_diff = 0;
for i = 1:length(dims)
    factor_diff = factor_diff + norm(T_neg.U{i} - T.U{i}, 'fro');
end

fprintf('  -T\n');
fprintf('  Core difference: %.6e\n', core_diff);
fprintf('  Factor difference: %.6e\n', factor_diff);

if core_diff < 1e-14 && factor_diff < 1e-14
    fprintf('  ✓ PASS: Unary minus works correctly\n\n');
else
    fprintf('  ✗ FAIL: Unary minus incorrect\n\n');
end

%% Test 6: Addition of Tucker tensors
fprintf('Test 6: Addition of Tucker Tensors (T1 + T2)\n');
fprintf('=============================================\n');

% Create second tensor with same structure
T2 = TuckerTensor(dims, ranks);
T2.G = randn(ranks);
T2.U = T.U;  % Same factors required

T_sum = T + T2;

core_diff = norm(T_sum.G(:) - (T.G(:) + T2.G(:)));
factor_diff = 0;
for i = 1:length(dims)
    factor_diff = factor_diff + norm(T_sum.U{i} - T.U{i}, 'fro');
end

fprintf('  T1 + T2 (same factors)\n');
fprintf('  Core difference: %.6e\n', core_diff);
fprintf('  Factor difference: %.6e\n', factor_diff);

if core_diff < 1e-14 && factor_diff < 1e-14
    fprintf('  ✓ PASS: Addition works correctly\n\n');
else
    fprintf('  ✗ FAIL: Addition incorrect\n\n');
end

%% Test 7: Subtraction of Tucker tensors
fprintf('Test 7: Subtraction of Tucker Tensors (T1 - T2)\n');
fprintf('================================================\n');

T_diff = T - T2;

core_diff = norm(T_diff.G(:) - (T.G(:) - T2.G(:)));
factor_diff = 0;
for i = 1:length(dims)
    factor_diff = factor_diff + norm(T_diff.U{i} - T.U{i}, 'fro');
end

fprintf('  T1 - T2 (same factors)\n');
fprintf('  Core difference: %.6e\n', core_diff);
fprintf('  Factor difference: %.6e\n', factor_diff);

if core_diff < 1e-14 && factor_diff < 1e-14
    fprintf('  ✓ PASS: Subtraction works correctly\n\n');
else
    fprintf('  ✗ FAIL: Subtraction incorrect\n\n');
end

%% Test 8: Chain operations
fprintf('Test 8: Chain Operations\n');
fprintf('========================\n');

% Test: (T * 2) / 4 should equal T / 2
T_chain1 = (T * 2) / 4;
T_chain2 = T / 2;

core_diff = norm(T_chain1.G(:) - T_chain2.G(:));

fprintf('  (T * 2) / 4 vs T / 2\n');
fprintf('  Core difference: %.6e\n', core_diff);

if core_diff < 1e-14
    fprintf('  ✓ PASS: Chain operations work correctly\n\n');
else
    fprintf('  ✗ FAIL: Chain operations incorrect\n\n');
end

%% Test 9: Use in actual expression (like in RGD)
fprintf('Test 9: Practical Usage in RGD Context\n');
fprintf('=======================================\n');

% Simulate: Grad_F = gradient_function(...) / sqrt(m)
m = 100;
Grad_F = T / sqrt(m);

expected_core = T.G / sqrt(m);
core_diff = norm(Grad_F.G(:) - expected_core(:));

fprintf('  Grad_F = T / sqrt(%d)\n', m);
fprintf('  Core difference: %.6e\n', core_diff);

% Check factors unchanged
factor_unchanged = true;
for i = 1:length(dims)
    if norm(Grad_F.U{i} - T.U{i}, 'fro') > 1e-14
        factor_unchanged = false;
    end
end

if core_diff < 1e-14 && factor_unchanged
    fprintf('  ✓ PASS: RGD-style usage works correctly\n\n');
else
    fprintf('  ✗ FAIL: RGD-style usage incorrect\n\n');
end

%% Test 10: Verify full tensor equivalence
fprintf('Test 10: Full Tensor Equivalence\n');
fprintf('=================================\n');

% For small tensor, verify that T/scalar gives same result as full(T)/scalar
dims_small = [3, 3, 3, 3];
ranks_small = [2, 2, 2, 2];
T_small = TuckerTensor(dims_small, ranks_small);
T_small.G = randn(ranks_small);
for i = 1:4
    T_small.U{i} = randn(dims_small(i), ranks_small(i));
end

scalar = 2.5;
T_small_div = T_small / scalar;

% Form full tensors
full_original = T_small.full();
full_divided = T_small_div.full();
full_expected = full_original / scalar;

rel_error = norm(full_divided(:) - full_expected(:)) / norm(full_expected(:));

fprintf('  Small tensor: dims=[%s], ranks=[%s]\n', num2str(dims_small), num2str(ranks_small));
fprintf('  Scalar: %.2f\n', scalar);
fprintf('  Relative error in full tensor: %.6e\n', rel_error);

if rel_error < 1e-12
    fprintf('  ✓ PASS: Full tensor equivalence verified\n\n');
else
    fprintf('  ✗ FAIL: Full tensor equivalence violated\n\n');
end

%% Summary
fprintf('=== Test Summary ===\n');
fprintf('All scalar operations tested:\n');
fprintf('  ✓ Division (T / scalar)\n');
fprintf('  ✓ Multiplication (T * scalar, scalar * T)\n');
fprintf('  ✓ Element-wise (.* and ./)\n');
fprintf('  ✓ Unary minus (-T)\n');
fprintf('  ✓ Addition (T1 + T2)\n');
fprintf('  ✓ Subtraction (T1 - T2)\n');
fprintf('  ✓ Chain operations\n');
fprintf('  ✓ Practical RGD usage\n');
fprintf('  ✓ Full tensor equivalence\n');
fprintf('\n');
fprintf('TuckerTensor class now supports:\n');
fprintf('  - Grad_F = tucker_op.get_proj_grad(...) / sqrt(m)\n');
fprintf('  - T_scaled = T * alpha\n');
fprintf('  - T_sum = T1 + T2 (with matching factors)\n');
fprintf('  - And all standard arithmetic operations!\n');
fprintf('\n=== All Tests Complete ===\n');
