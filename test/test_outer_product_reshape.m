%% Test: Verify reshape(B1(:) * B2(:)', [r,r,r,r]) gives dG(i,j,k,l) = B1(i,j) * B2(k,l)
% 
% This test checks if the vectorized outer product correctly computes
% the 4D tensor outer product of two matrices.

clear; clc;

fprintf('Testing: dG(i,j,k,l) = B1(i,j) * B2(k,l)\n');
fprintf('========================================\n\n');

%% Test 1: Small matrices (r=2)
fprintf('Test 1: r = 2\n');
r = 2;
B1 = randn(r, r);
B2 = randn(r, r);

% Method 1: Explicit nested loops
dG_loop = zeros(r, r, r, r);
for i = 1:r
    for j = 1:r
        for k = 1:r
            for l = 1:r
                dG_loop(i, j, k, l) = B1(i, j) * B2(k, l);
            end
        end
    end
end

% Method 2: Vectorized reshape (the one we're testing)
dG_reshape = reshape(B1(:) * B2(:)', [r, r, r, r]);

% Compare
diff = norm(dG_loop(:) - dG_reshape(:));
fprintf('  Max difference: %.15e\n', diff);
if diff < 1e-14
    fprintf('  ✓ PASSED\n\n');
else
    fprintf('  ✗ FAILED\n\n');
end

%% Test 2: Larger matrices (r=3)
fprintf('Test 2: r = 3\n');
r = 3;
B1 = randn(r, r);
B2 = randn(r, r);

% Method 1: Explicit nested loops
dG_loop = zeros(r, r, r, r);
for i = 1:r
    for j = 1:r
        for k = 1:r
            for l = 1:r
                dG_loop(i, j, k, l) = B1(i, j) * B2(k, l);
            end
        end
    end
end

% Method 2: Vectorized reshape
dG_reshape = reshape(B1(:) * B2(:)', [r, r, r, r]);

% Compare
diff = norm(dG_loop(:) - dG_reshape(:));
fprintf('  Max difference: %.15e\n', diff);
if diff < 1e-14
    fprintf('  ✓ PASSED\n\n');
else
    fprintf('  ✗ FAILED\n\n');
end

%% Test 3: Edge case r=1 (scalar case)
fprintf('Test 3: r = 1 (scalar case)\n');
r = 1;
B1 = randn(1, 1);
B2 = randn(1, 1);

% Method 1: Direct computation
dG_direct = B1 * B2;

% Method 2: Vectorized reshape
dG_reshape = reshape(B1(:) * B2(:)', [r, r, r, r]);

% For r=1, result should be a scalar
if isscalar(dG_reshape)
    fprintf('  Warning: MATLAB squeezed dimensions to scalar\n');
    fprintf('  dG_reshape is scalar: %.6f\n', dG_reshape);
    fprintf('  Expected value: %.6f\n', dG_direct);
else
    fprintf('  dG_reshape size: [%s]\n', num2str(size(dG_reshape)));
    fprintf('  dG_reshape value: %.6f\n', dG_reshape(1,1,1,1));
    fprintf('  Expected value: %.6f\n', dG_direct);
end

diff = abs(dG_reshape(1) - dG_direct);
fprintf('  Difference: %.15e\n', diff);
if diff < 1e-14
    fprintf('  ✓ PASSED (but dimensions may be squeezed)\n\n');
else
    fprintf('  ✗ FAILED\n\n');
end

%% Test 4: Verify specific indexing pattern
fprintf('Test 4: Verify indexing pattern\n');
r = 3;
B1 = magic(r);  % Use magic square for distinctive values
B2 = pascal(r); % Use Pascal matrix

% Compute using reshape
dG_reshape = reshape(B1(:) * B2(:)', [r, r, r, r]);

% Check a few specific elements
test_indices = {[1,1,1,1], [1,2,3,2], [2,2,2,2], [3,1,2,3]};
all_passed = true;

for t = 1:length(test_indices)
    idx = test_indices{t};
    i = idx(1); j = idx(2); k = idx(3); l = idx(4);
    
    expected = B1(i, j) * B2(k, l);
    actual = dG_reshape(i, j, k, l);
    diff = abs(expected - actual);
    
    fprintf('  dG(%d,%d,%d,%d): expected=%.6f, actual=%.6f, diff=%.2e', ...
            i, j, k, l, expected, actual, diff);
    
    if diff < 1e-14
        fprintf(' ✓\n');
    else
        fprintf(' ✗\n');
        all_passed = false;
    end
end

if all_passed
    fprintf('  ✓ All specific indices PASSED\n\n');
else
    fprintf('  ✗ Some indices FAILED\n\n');
end

%% Test 5: Alternative interpretation check
% Make sure B1(:) * B2(:)' is outer product, not inner product
fprintf('Test 5: Verify outer product vs inner product\n');
r = 2;
B1 = [1, 2; 3, 4];
B2 = [5, 6; 7, 8];

% Outer product: B1(:) * B2(:)' should be 4×4
outer = B1(:) * B2(:)';
fprintf('  B1(:) * B2(:)'' size: [%s] (should be [4, 4])\n', num2str(size(outer)));

% Inner product: B1(:)' * B2(:) would be scalar
inner = B1(:)' * B2(:);
fprintf('  B1(:)'' * B2(:) size: [%s] (should be [1, 1])\n', num2str(size(inner)));

if isequal(size(outer), [4, 4])
    fprintf('  ✓ Correct: Using outer product\n\n');
else
    fprintf('  ✗ ERROR: Not outer product!\n\n');
end

%% Summary
fprintf('========================================\n');
fprintf('CONCLUSION: reshape(B1(:) * B2(:)'', [r,r,r,r])\n');
fprintf('correctly computes dG(i,j,k,l) = B1(i,j) * B2(k,l)\n');
fprintf('========================================\n');
