% test_tensor_U_GD.m
% Test gradient descent on U for tensor-lifted problem

clear; clc;
addpath('../solver');
addpath('../utilities');

fprintf('=== Testing Tensor U Gradient Descent ===\n\n');

%% Problem Setup
d = 20;              % Matrix dimension
r = 1;               % Rank
m = 100;             % Number of measurements
T = 100;             % Iterations
mu = 0.05;           % Step size

% rng(42);

% Ground truth
U_true = randn(d, r);
Xstar = U_true * U_true';
Xstar = Xstar / norm(Xstar, 'fro');

fprintf('Setup: d=%d, r=%d, m=%d\n', d, r, m);
fprintf('Ground truth norm: %.6f\n\n', norm(Xstar, 'fro'));

%% Generate Measurements
fprintf('Generating measurements...\n');
A_cells = cell(m, 1);
y = zeros(m, 1);

for i = 1:m
    Ai = randn(d, d);
    Ai = (Ai + Ai') / 2;  % Symmetrize
    A_cells{i} = Ai;
    
    % Compute yᵢ = ⟨Aᵢ⊗Aᵢ, Xstar⊗Xstar⟩ = trace(AᵢXstarAᵢXstar)
    temp = Ai * Xstar;
    y(i) = trace(temp * temp)/ sqrt(m);
end

fprintf('Measurements generated: %d\n', m);
fprintf('Measurement range: [%.2e, %.2e]\n\n', min(y), max(y));

%% Initialize
fprintf('Initializing...\n');
U0 = randn(d, r) * 0.1;
X0 = U0 * U0';
X0 = (X0 + X0') / 2;
[init_error, ~] = rectify_sign_ambiguity(X0, Xstar);
fprintf('Initial error: %.6e\n\n', init_error);

%% Run Solver
fprintf('Running Tensor U Gradient Descent...\n');
params = struct();
params.T = T;
params.mu = mu;
params.Xstar = Xstar;
params.verbose = 1;

tic;
[U_final, X_final, history] = solve_tensor_U_GD(y, A_cells, U0, params);
time_elapsed = toc;

fprintf('\n');
[final_error, X_corrected] = rectify_sign_ambiguity(X_final, Xstar);
fprintf('Final error: %.6e\n', final_error);
fprintf('Time: %.2f seconds\n', time_elapsed);
fprintf('Final loss: %.6e\n\n', history.loss(end));

%% Convergence Plot
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
semilogy(1:T, history.loss, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Loss: ||y - A(T)||²');
title('Loss Convergence');
grid on;

subplot(1, 3, 2);
semilogy(1:T, history.errors, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Relative Error');
title('Error vs Ground Truth');
grid on;

subplot(1, 3, 3);
imagesc([Xstar, X_corrected]);
colorbar;
title('Ground Truth (left) vs Recovered (right)');
axis equal tight;

sgtitle(sprintf('Tensor U-GD: d=%d, r=%d, m=%d, final error=%.2e', d, r, m, final_error));

%% Summary
fprintf('=== Summary ===\n');
fprintf('Method: Gradient Descent on U directly\n');
fprintf('Formulation: X = UU^T, T = X ⊗ X\n');
fprintf('Loss: ||y - A(T)||²\n');
fprintf('Initial error: %.6e\n', init_error);
fprintf('Final error: %.6e\n', final_error);
fprintf('Improvement: %.2fx\n', init_error / final_error);
fprintf('Convergence: %s\n', ternary(final_error < 1e-6, 'Success', 'Incomplete'));
fprintf('\n✓ Test Complete\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
