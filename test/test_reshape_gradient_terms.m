%% Test: Verify Reshape Operations in Gradient Computation
% 
% This test verifies that the efficient reshape operations correctly compute
% the mode-unfolded gradient terms for the Tucker manifold projection.
%
% For each mode k, we verify:
%   reshape((reshape(B1, [d*r,1]) * reshape(B2, [1, r*r])), [d, r^3])
% produces the correct mode-k unfolding of (A_i ⊗ A_i) ×_{j≠k} U_j^T

clear; clc;

fprintf('=== Testing Reshape Operations in Gradient Computation ===\n\n');

%% Test Parameters
d = 6;    % Dimension (keep small for explicit verification)
r = 3;    % Tucker rank

fprintf('Test setup: d=%d, r=%d\n\n', d, r);

%% Create Test Data
fprintf('Creating test data...\n');

% Random measurement matrix
Ai = randn(d, d);
Ai = (Ai + Ai') / 2;  % Make symmetric

% Random factor matrices (orthogonal)
U1 = orth(randn(d, r));
U2 = orth(randn(d, r));
U3 = orth(randn(d, r));
U4 = orth(randn(d, r));

% Test residual
residual = randn(1);

fprintf('  A_i: %dx%d (symmetric)\n', d, d);
fprintf('  U_k: %dx%d (orthogonal)\n', d, r);
fprintf('  residual: %.6f\n\n', residual);

%% Test Mode 1: grad_term_1
fprintf('--- Test Mode 1: Computing grad_term_1 ---\n');
fprintf('Should compute: (A_i ⊗ A_i) ×₂ U₂^T ×₃ U₃^T ×₄ U₄^T in mode-1 unfolding\n\n');

% Method 1: Efficient reshape (current implementation)
fprintf('Method 1 (Efficient Reshape):\n');
B1 = Ai * U2;  % d × r
B2 = U3' * Ai * U4;  % r × r
grad_term_1_efficient = residual * reshape((reshape(B1, [d*r,1]) * reshape(B2, [1, r*r])), [d, r^3]);
fprintf('  Computed grad_term_1: %dx%d\n', size(grad_term_1_efficient, 1), size(grad_term_1_efficient, 2));
fprintf('  Norm: %.6e\n', norm(grad_term_1_efficient(:)));

% Method 2: Explicit full tensor computation (reference)
fprintf('\nMethod 2 (Explicit Full Tensor):\n');
grad_term_1_explicit = zeros(d, r^3);

% Form A_i ⊗ A_i as 4D tensor
AiAi = zeros(d, d, d, d);
for i1 = 1:d
    for i2 = 1:d
        for i3 = 1:d
            for i4 = 1:d
                AiAi(i1, i2, i3, i4) = Ai(i1, i2) * Ai(i3, i4);
            end
        end
    end
end

% Contract with U2, U3, U4 and unfold along mode 1
idx = 1;
for j2 = 1:r
    for j3 = 1:r
        for j4 = 1:r
            for i1 = 1:d
                for i2 = 1:d
                    for i3 = 1:d
                        for i4 = 1:d
                            grad_term_1_explicit(i1, idx) = grad_term_1_explicit(i1, idx) + ...
                                residual * AiAi(i1, i2, i3, i4) * U2(i2, j2) * U3(i3, j3) * U4(i4, j4);
                        end
                    end
                end
            end
            idx = idx + 1;
        end
    end
end
fprintf('  Computed grad_term_1: %dx%d\n', size(grad_term_1_explicit, 1), size(grad_term_1_explicit, 2));
fprintf('  Norm: %.6e\n', norm(grad_term_1_explicit(:)));

% Compare
error_mode1 = norm(grad_term_1_efficient - grad_term_1_explicit) / norm(grad_term_1_explicit);
fprintf('\nComparison:\n');
fprintf('  Relative error: %.6e\n', error_mode1);
if error_mode1 < 1e-12
    fprintf('  ✓ PASS: Mode 1 gradient term computed correctly\n\n');
else
    fprintf('  ✗ FAIL: Mode 1 gradient term has errors!\n\n');
end

%% Test Mode 2: grad_term_2
fprintf('--- Test Mode 2: Computing grad_term_2 ---\n');
fprintf('Should compute: (A_i ⊗ A_i) ×₁ U₁^T ×₃ U₃^T ×₄ U₄^T in mode-2 unfolding\n\n');

% Method 1: Efficient reshape
fprintf('Method 1 (Efficient Reshape):\n');
B1_m2 = Ai' * U1;  % d × r
B2_m2 = U3' * Ai * U4;  % r × r
grad_term_2_efficient = residual * reshape((reshape(B1_m2, [d*r, 1]) * reshape(B2_m2, [1, r*r])), [d, r^3]);
fprintf('  Computed grad_term_2: %dx%d\n', size(grad_term_2_efficient, 1), size(grad_term_2_efficient, 2));
fprintf('  Norm: %.6e\n', norm(grad_term_2_efficient(:)));

% Method 2: Explicit
fprintf('\nMethod 2 (Explicit Full Tensor):\n');
grad_term_2_explicit = zeros(d, r^3);
idx = 1;
for j1 = 1:r
    for j3 = 1:r
        for j4 = 1:r
            for i2 = 1:d
                for i1 = 1:d
                    for i3 = 1:d
                        for i4 = 1:d
                            grad_term_2_explicit(i2, idx) = grad_term_2_explicit(i2, idx) + ...
                                residual * AiAi(i1, i2, i3, i4) * U1(i1, j1) * U3(i3, j3) * U4(i4, j4);
                        end
                    end
                end
            end
            idx = idx + 1;
        end
    end
end
fprintf('  Computed grad_term_2: %dx%d\n', size(grad_term_2_explicit, 1), size(grad_term_2_explicit, 2));
fprintf('  Norm: %.6e\n', norm(grad_term_2_explicit(:)));

error_mode2 = norm(grad_term_2_efficient - grad_term_2_explicit) / norm(grad_term_2_explicit);
fprintf('\nComparison:\n');
fprintf('  Relative error: %.6e\n', error_mode2);
if error_mode2 < 1e-12
    fprintf('  ✓ PASS: Mode 2 gradient term computed correctly\n\n');
else
    fprintf('  ✗ FAIL: Mode 2 gradient term has errors!\n\n');
end

%% Test Mode 3: grad_term_3
fprintf('--- Test Mode 3: Computing grad_term_3 ---\n');
fprintf('Should compute: (A_i ⊗ A_i) ×₁ U₁^T ×₂ U₂^T ×₄ U₄^T in mode-3 unfolding\n\n');

% Method 1: Efficient reshape
fprintf('Method 1 (Efficient Reshape):\n');
B1_m3 = Ai * U4;  % d × r
B2_m3 = U1' * Ai * U2;  % r × r
grad_term_3_efficient = residual * reshape((reshape(B1_m3, [d*r, 1]) * reshape(B2_m3, [1, r*r])), [d, r^3]);
fprintf('  Computed grad_term_3: %dx%d\n', size(grad_term_3_efficient, 1), size(grad_term_3_efficient, 2));
fprintf('  Norm: %.6e\n', norm(grad_term_3_efficient(:)));

% Method 2: Explicit
fprintf('\nMethod 2 (Explicit Full Tensor):\n');
grad_term_3_explicit = zeros(d, r^3);
idx = 1;
for j1 = 1:r
    for j2 = 1:r
        for j4 = 1:r
            for i3 = 1:d
                for i1 = 1:d
                    for i2 = 1:d
                        for i4 = 1:d
                            grad_term_3_explicit(i3, idx) = grad_term_3_explicit(i3, idx) + ...
                                residual * AiAi(i1, i2, i3, i4) * U1(i1, j1) * U2(i2, j2) * U4(i4, j4);
                        end
                    end
                end
            end
            idx = idx + 1;
        end
    end
end
fprintf('  Computed grad_term_3: %dx%d\n', size(grad_term_3_explicit, 1), size(grad_term_3_explicit, 2));
fprintf('  Norm: %.6e\n', norm(grad_term_3_explicit(:)));

error_mode3 = norm(grad_term_3_efficient - grad_term_3_explicit) / norm(grad_term_3_explicit);
fprintf('\nComparison:\n');
fprintf('  Relative error: %.6e\n', error_mode3);
if error_mode3 < 1e-12
    fprintf('  ✓ PASS: Mode 3 gradient term computed correctly\n\n');
else
    fprintf('  ✗ FAIL: Mode 3 gradient term has errors!\n\n');
end

%% Test Mode 4: grad_term_4
fprintf('--- Test Mode 4: Computing grad_term_4 ---\n');
fprintf('Should compute: (A_i ⊗ A_i) ×₁ U₁^T ×₂ U₂^T ×₃ U₃^T in mode-4 unfolding\n\n');

% Method 1: Efficient reshape
fprintf('Method 1 (Efficient Reshape):\n');
B1_m4 = Ai' * U3;  % d × r
B2_m4 = U1' * Ai * U2;  % r × r
grad_term_4_efficient = residual * reshape((reshape(B1_m4, [d*r, 1]) * reshape(B2_m4, [1, r*r])), [d, r^3]);
fprintf('  Computed grad_term_4: %dx%d\n', size(grad_term_4_efficient, 1), size(grad_term_4_efficient, 2));
fprintf('  Norm: %.6e\n', norm(grad_term_4_efficient(:)));

% Method 2: Explicit
fprintf('\nMethod 2 (Explicit Full Tensor):\n');
grad_term_4_explicit = zeros(d, r^3);
idx = 1;
for j1 = 1:r
    for j2 = 1:r
        for j3 = 1:r
            for i4 = 1:d
                for i1 = 1:d
                    for i2 = 1:d
                        for i3 = 1:d
                            grad_term_4_explicit(i4, idx) = grad_term_4_explicit(i4, idx) + ...
                                residual * AiAi(i1, i2, i3, i4) * U1(i1, j1) * U2(i2, j2) * U3(i3, j3);
                        end
                    end
                end
            end
            idx = idx + 1;
        end
    end
end
fprintf('  Computed grad_term_4: %dx%d\n', size(grad_term_4_explicit, 1), size(grad_term_4_explicit, 2));
fprintf('  Norm: %.6e\n', norm(grad_term_4_explicit(:)));

error_mode4 = norm(grad_term_4_efficient - grad_term_4_explicit) / norm(grad_term_4_explicit);
fprintf('\nComparison:\n');
fprintf('  Relative error: %.6e\n', error_mode4);
if error_mode4 < 1e-12
    fprintf('  ✓ PASS: Mode 4 gradient term computed correctly\n\n');
else
    fprintf('  ✗ FAIL: Mode 4 gradient term has errors!\n\n');
end

%% Test Intermediate Outer Product Operation
fprintf('--- Test: Verifying Outer Product Operation ---\n');
fprintf('Testing: reshape(B1, [d*r,1]) * reshape(B2, [1, r*r])\n');
fprintf('Should produce: (d*r) × (r*r) matrix\n\n');

% Use Mode 1 data
B1_vec = reshape(B1, [d*r, 1]);
B2_vec = reshape(B2, [1, r*r]);
outer_product = B1_vec * B2_vec;

fprintf('B1: %dx%d → B1_vec: %dx%d\n', size(B1, 1), size(B1, 2), size(B1_vec, 1), size(B1_vec, 2));
fprintf('B2: %dx%d → B2_vec: %dx%d\n', size(B2, 1), size(B2, 2), size(B2_vec, 1), size(B2_vec, 2));
fprintf('Outer product: %dx%d\n', size(outer_product, 1), size(outer_product, 2));
fprintf('Expected: %dx%d\n', d*r, r*r);

if isequal(size(outer_product), [d*r, r*r])
    fprintf('✓ PASS: Outer product has correct dimensions\n\n');
else
    fprintf('✗ FAIL: Outer product has wrong dimensions!\n\n');
end

% Verify reshaping to [d, r^3]
reshaped = reshape(outer_product, [d, r^3]);
fprintf('After reshape to [d, r^3]: %dx%d\n', size(reshaped, 1), size(reshaped, 2));
fprintf('Expected: %dx%d\n', d, r^3);

if isequal(size(reshaped), [d, r^3])
    fprintf('✓ PASS: Final reshape produces correct dimensions\n\n');
else
    fprintf('✗ FAIL: Final reshape has wrong dimensions!\n\n');
end

%% Test Element-wise Verification
fprintf('--- Test: Element-wise Verification (Mode 1) ---\n');
fprintf('Checking a few random elements...\n\n');

% Pick random elements to verify
test_elements = [1, 1; d, 1; ceil(d/2), ceil(r^3/2); d, r^3; 1, r^3];
all_match = true;

for t = 1:size(test_elements, 1)
    i1_test = test_elements(t, 1);
    col_test = test_elements(t, 2);
    
    val_efficient = grad_term_1_efficient(i1_test, col_test);
    val_explicit = grad_term_1_explicit(i1_test, col_test);
    diff = abs(val_efficient - val_explicit);
    
    fprintf('  Element (%d, %d): efficient=%.6e, explicit=%.6e, diff=%.2e', ...
            i1_test, col_test, val_efficient, val_explicit, diff);
    
    if diff < 1e-12 * max([abs(val_efficient), abs(val_explicit), 1e-10])
        fprintf(' ✓\n');
    else
        fprintf(' ✗\n');
        all_match = false;
    end
end

if all_match
    fprintf('\n✓ All sampled elements match\n\n');
else
    fprintf('\n✗ Some elements do not match\n\n');
end

%% Summary
fprintf('=== Test Summary ===\n');
max_error = max([error_mode1, error_mode2, error_mode3, error_mode4]);
fprintf('Maximum relative error across all modes: %.6e\n', max_error);

if max_error < 1e-12
    fprintf('\n✓ ALL TESTS PASSED\n');
    fprintf('The reshape operations correctly compute mode-unfolded gradient terms.\n');
    fprintf('\nFormula verified for all modes:\n');
    fprintf('  grad_term_k = residual * reshape((reshape(B1, [d*r,1]) * reshape(B2, [1,r*r])), [d, r^3])\n');
else
    fprintf('\n✗ SOME TESTS FAILED\n');
    fprintf('There may be an error in the reshape operations!\n');
end

fprintf('\nNote: This efficient formulation avoids forming the full (A_i ⊗ A_i)\n');
fprintf('tensor, which would be d×d×d×d = %d^4 = %d elements.\n', d, d^4);
fprintf('Instead we work with matrices of size (d×r) and (r×r), total %d + %d elements.\n', ...
        d*r, r*r);
fprintf('Memory savings: %.1fx\n', d^4 / (d*r + r*r));
