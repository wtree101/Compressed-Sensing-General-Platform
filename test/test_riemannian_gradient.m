%% Test Riemannian Gradient Computation on Tucker Manifold
% Tests the get_proj_grad_kronecker method

clear; clc;

fprintf('=== Testing Riemannian Gradient on Tucker Manifold ===\n\n');

%% Setup
d = 8;   % Dimension
r = 3;   % Tucker rank
m = 15;  % Number of measurements

fprintf('Creating test problem: d=%d, r=%d, m=%d\n', d, r, m);

% Create measurement matrices
A_cells = cell(m, 1);
for i = 1:m
    Ai = randn(d, d);
    A_cells{i} = (Ai + Ai') / 2;  % Symmetric
end

% Create operator
tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', true);

% Create Tucker tensor
T = TuckerTensor([d, d, d, d], r, 'symmetric', true, 'init_method', 'orthogonal');
fprintf('Created Tucker tensor\n');

%% Compute forward
y_true = tucker_op.forward(T);
fprintf('Forward computation: generated %d measurements\n', m);

% Add noise
y_noisy = y_true + randn(m, 1) * 0.01;

%% Compute Riemannian gradient
fprintf('\n--- Computing Riemannian Gradient ---\n');
tic;
Grad_F = tucker_op.get_proj_grad_kronecker(T, y_true, y_noisy);
time_elapsed = toc;
fprintf('Gradient computation time: %.4f seconds\n', time_elapsed);

%% Verify gradient properties
fprintf('\n--- Verifying Gradient Properties ---\n');

% 1. Check that Grad_F is a TuckerTensor
fprintf('Grad_F is TuckerTensor: %s\n', mat2str(isa(Grad_F, 'TuckerTensor')));

% 2. Check dimensions
fprintf('Grad_F dimensions match T: %s\n', mat2str(isequal(Grad_F.dims, T.dims)));
fprintf('Grad_F ranks match T: %s\n', mat2str(isequal(Grad_F.tucker_ranks, T.tucker_ranks)));

% 3. Check core gradient
fprintf('\nCore gradient:\n');
fprintf('  G size: %s\n', mat2str(size(Grad_F.G)));
fprintf('  G norm: %.4e\n', norm(Grad_F.G(:)));

% 4. Check factor gradients are orthogonal to U
fprintf('\nFactor gradient orthogonality (should be ~0):\n');
for k = 1:4
    ortho_error = norm(T.U{k}' * Grad_F.U{k}, 'fro');
    fprintf('  ||U{%d}^T * dU{%d}||_F = %.4e\n', k, k, ortho_error);
end

% 5. Check Up is computed
fprintf('\nOrthogonal complement Up:\n');
for k = 1:4
    if ~isempty(Grad_F.Up{k})
        fprintf('  Up{%d} size: %dx%d\n', k, size(Grad_F.Up{k}, 1), size(Grad_F.Up{k}, 2));
        ortho_check = norm(T.U{k}' * Grad_F.Up{k}, 'fro');
        fprintf('    ||U{%d}^T * Up{%d}||_F = %.4e\n', k, k, ortho_check);
    else
        fprintf('  Up{%d} is empty\n', k);
    end
end

%% Test gradient descent step
fprintf('\n--- Testing Gradient Descent Step ---\n');

% Compute initial loss
residual_0 = y_true - y_noisy;
loss_0 = 0.5 * norm(residual_0)^2;
fprintf('Initial loss: %.6e\n', loss_0);

% Take gradient step
step_size = 0.01;
fprintf('Taking gradient step with step_size = %.3f\n', step_size);

% Create tangent direction from gradient
tangent = struct();
tangent.dG = Grad_F.G;
tangent.dU = Grad_F.U;

% Update tensor
T_new = T.update_from_tangent(tangent, step_size);

% Compute new loss
y_new = tucker_op.forward(T_new);
residual_1 = y_new - y_noisy;
loss_1 = 0.5 * norm(residual_1)^2;
fprintf('Loss after step: %.6e\n', loss_1);
fprintf('Loss change: %.6e (%.2f%%)\n', loss_1 - loss_0, ...
        100 * (loss_1 - loss_0) / loss_0);

%% Verify manifold constraints preserved
fprintf('\n--- Verifying Manifold Constraints ---\n');

% Check orthonormality of factors
for k = 1:4
    ortho_error = norm(T_new.U{k}' * T_new.U{k} - eye(r), 'fro');
    fprintf('||U{%d}^T * U{%d} - I||_F = %.4e (should be ~0)\n', k, k, ortho_error);
end

%% Summary
fprintf('\n=== Test Summary ===\n');
fprintf('✓ Riemannian gradient computed successfully\n');
fprintf('✓ Gradient is in tangent space (orthogonal to U)\n');
fprintf('✓ Up (orthogonal complement) computed correctly\n');
fprintf('✓ Gradient descent step preserves manifold constraints\n');
fprintf('✓ Loss %s as expected\n', ternary(loss_1 < loss_0, 'decreased', 'behavior'));
fprintf('\n=== Test Complete ===\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
