%% Test Tucker Tensor Lift Initialization
% Test Tucker-based tensor lift (non-symmetric, debug mode)

clear; clc;

fprintf('=== Tucker Tensor Lift Test (Non-Symmetric, Debug Mode) ===\n\n');

%% Setup
d = 20; r = 1; r_tucker = r; m = 500; mu = 0.005; T_power = 5;
rng(42);

% Ground truth
U_true = randn(d, r);
Xstar = U_true * U_true';
Xstar = Xstar / norm(Xstar, 'fro');

% Measurements: y_i = <A_i ⊗ A_i, X ⊗ X>
A_matrix = randn(m, d*d);
y = zeros(m, 1);
for i = 1:m
    Ai = reshape(A_matrix(i, :), [d, d]);
    Ai = (Ai + Ai') / 2;
    y(i) = trace(Ai * Xstar * Ai * Xstar);
end
y = y / sqrt(m);

operator = struct();
operator.A = @(X) forward_op(X, A_matrix, d);

fprintf('Setup: d=%d, r=%d, r_tucker=%d, m=%d\n\n', d, r, r_tucker, m);

%% Test 1: Basic Tucker Initialization (Debug Mode)
fprintf('=== Test 1: Tucker RGD Initialization (Debug Mode) ===\n');

params = struct('T_power', T_power, 'mu', mu, 'r', r, ...
                'Xstar', Xstar, 'verbose', 1, 'symmetric', true, ...
                'debug', true);  % Enable debug mode to track tensor errors

tic;
[X_tucker, ~, hist] = initialize_tensor_lift_tucker(y, operator, d, d, params);
time_tucker = toc;

[error_tucker, ~] = rectify_sign_ambiguity(X_tucker, Xstar);

fprintf('\nResults: Error=%.2e, Time=%.2fs, Memory=%.2fKB\n', ...
        error_tucker, time_tucker, (4*d*r_tucker + r_tucker^4)*8/1024);

if isfield(hist, 'tensor_errors_relative')
    fprintf('[DEBUG] Final tensor error (relative): %.2e\n', hist.tensor_errors_relative(end));
end
fprintf('\n');


%% Test 3: Convergence Plot
fprintf('=== Test 3: Convergence Visualization ===\n');

figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
semilogy(1:length(hist.loss_function), hist.loss_function, 'b-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Loss'); title('Loss Convergence'); grid on;

subplot(1, 3, 2);
semilogy(1:length(hist.matrix_errors), hist.matrix_errors, 'r-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Matrix Error'); title('Matrix Error vs Ground Truth'); grid on;

if isfield(hist, 'tensor_errors_relative')
    subplot(1, 3, 3);
    semilogy(1:length(hist.tensor_errors_relative), hist.tensor_errors_relative, 'g-', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Tensor Error (Relative)'); 
    title('Tensor Error: ||T - T*||/||T*||'); grid on;
else
    subplot(1, 3, 3);
    text(0.5, 0.5, 'Tensor error tracking disabled', 'HorizontalAlignment', 'center');
    axis off;
end

sgtitle('Tucker Tensor Lift Convergence (Debug Mode)');

fprintf('Convergence plots generated.\n\n');

%% Summary
fprintf('=== Summary ===\n');
full_mem = d^4 * 8;
tucker_mem = (4*d*r_tucker + r_tucker^4) * 8;
fprintf('Memory: Full tensor=%.1fMB, Tucker=%.1fKB (%.0fx compression)\n', ...
        full_mem/1024/1024, tucker_mem/1024, full_mem/tucker_mem);
fprintf('Final error: %.2e (in %d iterations)\n', error_tucker, T_power);
fprintf('Tucker format: 4 independent factors U₁,U₂,U₃,U₄ (non-symmetric)\n');
if isfield(hist, 'tensor_errors_relative')
    fprintf('[DEBUG] Tensor error enabled: tracks ||T-T*||/||T*|| convergence\n');
end
fprintf('\n✓ Test Complete\n');

%% Helper function
function y = forward_op(X, A_matrix, d)
    % Forward operator: y = A * vec(X ⊗ X)
    m = size(A_matrix, 1);
    y = zeros(m, 1);
    for i = 1:m
        Ai = reshape(A_matrix(i, :), [d, d]);
        Ai = (Ai + Ai') / 2;
        y(i) = trace(Ai * X * Ai * X);
    end
end

