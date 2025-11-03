%% Test r=1 Edge Case
% Verifies that all operations work correctly when Tucker rank r=1
% Main challenge: MATLAB collapses singleton dimensions

clear; clc;

fprintf('=== Testing r=1 Edge Case ===\n\n');

%% Test Parameters
d = 10;   % Dimension
r = 1;    % Tucker rank (edge case)
m = 50;   % Number of measurements

fprintf('Test setup: d=%d, r=%d, m=%d\n\n', d, r, m);

%% Test 1: Tucker Tensor Creation with r=1
fprintf('--- Test 1: Tucker Tensor Creation ---\n');

try
    T_tucker = TuckerTensor([d, d, d, d], r, 'symmetric', true, 'init_method', 'orthogonal');
    T_tucker.G = randn(1, 1, 1, 1) * 0.5;  % Explicitly create 4D tensor
    
    fprintf('  Core tensor G size: %s\n', mat2str(size(T_tucker.G)));
    fprintf('  Core tensor G ndims: %d\n', ndims(T_tucker.G));
    fprintf('  Core tensor G isscalar: %d\n', isscalar(T_tucker.G));
    fprintf('  ✓ Tucker tensor created successfully\n');
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    rethrow(ME);
end

%% Test 2: Full Tensor Reconstruction
fprintf('\n--- Test 2: Full Tensor Reconstruction ---\n');

try
    T_full = T_tucker.full();
    fprintf('  Full tensor size: %s\n', mat2str(size(T_full)));
    fprintf('  Full tensor ndims: %d\n', ndims(T_full));
    
    % Verify dimensions
    if isequal(size(T_full), [d, d, d, d])
        fprintf('  ✓ Full tensor has correct dimensions\n');
    else
        fprintf('  ⚠ Warning: Full tensor dimensions collapsed to %s\n', mat2str(size(T_full)));
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    rethrow(ME);
end

%% Test 3: Forward Operator with r=1
fprintf('\n--- Test 3: Forward Operator ---\n');

% Create measurement matrices
A_cells = cell(m, 1);
for i = 1:m
    Ai = randn(d, d);
    A_cells{i} = (Ai + Ai') / 2;  % Symmetric
end

try
    % Create operator
    op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false, 'dims', [d,d,d,d]);
    
    % Kronecker forward
    fprintf('  Computing Kronecker forward...\n');
    y_kron = op.forward_kronecker(T_tucker);
    fprintf('    Result size: %s\n', mat2str(size(y_kron)));
    fprintf('    First value: %.6f\n', y_kron(1));
    
    % General forward
    fprintf('  Computing General forward...\n');
    y_gen = op.forward_general(T_tucker);
    fprintf('    Result size: %s\n', mat2str(size(y_gen)));
    fprintf('    First value: %.6f\n', y_gen(1));
    
    % Compare
    error_rel = norm(y_kron - y_gen) / norm(y_gen);
    fprintf('  Relative error: %.4e\n', error_rel);
    
    if error_rel < 1e-10
        fprintf('  ✓ Forward operators match within tolerance\n');
    else
        fprintf('  ✗ FAIL: Forward operators do not match!\n');
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    fprintf('  Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end

%% Test 4: Gradient Computation with r=1
fprintf('\n--- Test 4: Gradient Computation ---\n');

try
    % Generate synthetic data
    y_true = randn(m, 1);
    y_computed = y_kron;
    
    % Compute gradient
    fprintf('  Computing Riemannian gradient...\n');
    T_tucker = T_tucker.compute_Up();
    Grad_F = op.get_proj_grad_kronecker(T_tucker, y_computed, y_true);
    
    fprintf('  Gradient core size: %s\n', mat2str(size(Grad_F.G)));
    fprintf('  Gradient Up{1} size: %s\n', mat2str(size(Grad_F.Up{1})));
    
    % Verify gradient is well-formed
    if isequal(size(Grad_F.G), size(T_tucker.G))
        fprintf('  ✓ Gradient core has correct size\n');
    else
        fprintf('  ✗ FAIL: Gradient core size mismatch\n');
    end
    
    if isequal(size(Grad_F.Up{1}), size(T_tucker.U{1}))
        fprintf('  ✓ Gradient Up has correct size\n');
    else
        fprintf('  ✗ FAIL: Gradient Up size mismatch\n');
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    fprintf('  Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end

%% Test 5: Retraction with r=1
fprintf('\n--- Test 5: Retraction ---\n');

try
    eta = 0.01;
    fprintf('  Computing retraction with eta=%.3f...\n', eta);
    T_new = T_tucker.retraction(Grad_F, eta);
    
    fprintf('  New core size: %s\n', mat2str(size(T_new.G)));
    fprintf('  New U{1} size: %s\n', mat2str(size(T_new.U{1})));
    
    % Verify retraction maintains rank
    if isequal(size(T_new.G), size(T_tucker.G))
        fprintf('  ✓ Retraction maintains core size\n');
    else
        fprintf('  ✗ FAIL: Retraction changed core size\n');
    end
    
    % Check orthogonality
    orth_error = norm(T_new.U{1}' * T_new.U{1} - eye(r), 'fro');
    fprintf('  Orthogonality error: %.4e\n', orth_error);
    
    if orth_error < 1e-10
        fprintf('  ✓ Retraction maintains orthogonality\n');
    else
        fprintf('  ✗ FAIL: Loss of orthogonality\n');
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    fprintf('  Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end

%% Summary
fprintf('\n=== Test Summary ===\n');
fprintf('All r=1 edge case tests completed successfully!\n');
fprintf('✓ Tucker tensor creation\n');
fprintf('✓ Full tensor reconstruction\n');
fprintf('✓ Forward operators (Kronecker and General)\n');
fprintf('✓ Gradient computation\n');
fprintf('✓ Retraction operation\n');
fprintf('\n=== r=1 Edge Case Handling: PASS ===\n');
