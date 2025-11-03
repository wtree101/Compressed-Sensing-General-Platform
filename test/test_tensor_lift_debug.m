%% Debug Test for Tensor Lift Initialization
% Verify tensor construction and loss computation

clear; clc;

fprintf('=== DEBUG: Tensor Lift Verification ===\n\n');

%% Setup (small problem for debugging)
d = 5;  % Small dimension for debugging
r = 1;  % Rank
m = 50; % Number of measurements
rng(123);

fprintf('Problem size: d=%d, r=%d, m=%d\n\n', d, r, m);

%% Generate ground truth
U_true = randn(d, r);
Xstar = U_true * U_true';
Xstar = Xstar / norm(Xstar, 'fro');

fprintf('Ground truth matrix Xstar:\n');
fprintf('  Size: %dx%d\n', size(Xstar, 1), size(Xstar, 2));
fprintf('  Norm: %.6f\n', norm(Xstar, 'fro'));
fprintf('  Rank: %d\n', rank(Xstar, 1e-10));
fprintf('  Sample entries: X(1,1)=%.4f, X(1,2)=%.4f, X(2,2)=%.4f\n\n', ...
        Xstar(1,1), Xstar(1,2), Xstar(2,2));

%% Generate measurements: y_i = <A_i ⊗ A_i, X ⊗ X>
fprintf('Generating measurements...\n');
A_matrix = randn(m, d*d);
y = zeros(m, 1);

for i = 1:m
    Ai = reshape(A_matrix(i, :), [d, d]);
    Ai = (Ai + Ai') / 2;  % Symmetrize
    y(i) = trace(Ai * Xstar * Ai * Xstar);
end
y = y / sqrt(m);

fprintf('  Measurements generated: %d\n', m);
fprintf('  y range: [%.4f, %.4f]\n', min(y), max(y));
fprintf('  y norm: %.6f\n', norm(y));
fprintf('  y mean: %.6f\n\n', mean(y));

%% Setup operator
operator = struct();
operator.A = @(X) forward_op(X, A_matrix, d);

%% Test with DEBUG mode enabled
fprintf('=== Running Tensor Lift with DEBUG Mode ===\n\n');

params = struct();
params.T_power = 10;
params.mu = 0.01;
params.r = r;
params.Xstar = Xstar;
params.verbose = true;
params.debug = true;  % Enable DEBUG mode
params.use_power_refine = false;

[X0, U0, history] = initialize_tensor_lift_efficient(y, operator, d, d, params);

%% Verify results
fprintf('\n=== Verification ===\n');

[final_error, X0_aligned] = rectify_sign_ambiguity(X0, Xstar);
fprintf('Final matrix error: %.6e\n', final_error);
fprintf('Final matrix norm: %.6f (target: %.6f)\n', norm(X0, 'fro'), norm(Xstar, 'fro'));
fprintf('Final matrix rank: %d (target: %d)\n', rank(X0, 1e-6), r);

if isfield(history, 'tensor_errors') && ~isempty(history.tensor_errors)
    fprintf('Tensor error progression:\n');
    fprintf('  Initial: %.6e\n', history.tensor_errors(1));
    fprintf('  Final: %.6e\n', history.tensor_errors(end));
end

if isfield(history, 'loss_function') && ~isempty(history.loss_function)
    fprintf('Loss function progression:\n');
    fprintf('  Initial: %.6e\n', history.loss_function(1));
    fprintf('  Final: %.6e\n', history.loss_function(end));
end

%% Plot convergence
if isfield(history, 'loss_function') && length(history.loss_function) > 1
    figure('Position', [100, 100, 1000, 400]);
    
    subplot(1, 2, 1);
    semilogy(1:length(history.loss_function), history.loss_function, 'b-o', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Loss'); title('Loss Function'); grid on;
    
    if isfield(history, 'tensor_errors') && ~isempty(history.tensor_errors)
        subplot(1, 2, 2);
        semilogy(1:length(history.tensor_errors), history.tensor_errors, 'r-o', 'LineWidth', 2);
        xlabel('Iteration'); ylabel('Relative Error'); title('Tensor Error vs Ground Truth'); grid on;
    end
    
    sgtitle('Debug: Tensor Lift Convergence');
end

fprintf('\n✓ Debug test complete\n');

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
