%%%%%%%%%% Test Tensor Lift Initialization
% Test tensor-lifted initialization for matrix recovery

clear; clc; close all;

fprintf('=== Tensor Lift Initialization Test ===\n\n');

%% Problem Setup
d = 20;  % Must be symmetric for tensor lift
d1 = d; d2 = d;
n = d * d;
m = 400;
rank_true = 2;

fprintf('Configuration: %dx%d symmetric matrix, rank=%d, m=%d measurements\n', d, d, rank_true, m);

%% Generate Symmetric Low-Rank Ground Truth
U_true = randn(d, rank_true);
Xstar = U_true * U_true';  % Symmetric rank-r matrix
Xstar = Xstar / norm(Xstar, 'fro');

fprintf('Ground truth rank: %d\n', rank(Xstar));
fprintf('Ground truth norm: %.6f\n', norm(Xstar, 'fro'));
fprintf('Symmetry error: %.2e\n\n', norm(Xstar - Xstar', 'fro'));

%% Generate Measurements
A = randn(m, n);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(z) reshape(A' * z, [d, d]);

% Phase retrieval: observe only magnitudes
y = abs(operator.A(Xstar)) / sqrt(m);

fprintf('Measurement range: [%.3f, %.3f]\n\n', min(y), max(y));

%% Test 1: Tensor Lift Initialization (Default T_tensor=5)
fprintf('--- Test 1: Tensor Lift (T_tensor=5) ---\n');
params1 = struct();
params1.r = rank_true;
params1.T_tensor = 5;
params1.Xstar = Xstar;
params1.projection = @(X) rank_projection(X, rank_true);
params1.verbose = true;

tic;
[X0_tensor5, U0_tensor5, history1] = initialize_tensor_lift(y, operator, d, d, params1);
t1 = toc;

[error1, X0_tensor5_aligned] = rectify_sign_ambiguity(X0_tensor5, Xstar);
rank1 = rank(X0_tensor5, 1e-6);

fprintf('Results:\n');
fprintf('  Final error: %.6e\n', error1);
fprintf('  Recovered rank: %d (true: %d)\n', rank1, rank_true);
fprintf('  Time: %.3f seconds\n\n', t1);

%% Test 2: Tensor Lift with More Iterations (T_tensor=10)
fprintf('--- Test 2: Tensor Lift (T_tensor=10) ---\n');
params2 = struct();
params2.r = rank_true;
params2.T_tensor = 10;
params2.Xstar = Xstar;
params2.projection = @(X) rank_projection(X, rank_true);
params2.verbose = true;

tic;
[X0_tensor10, U0_tensor10, history2] = initialize_tensor_lift(y, operator, d, d, params2);
t2 = toc;

[error2, X0_tensor10_aligned] = rectify_sign_ambiguity(X0_tensor10, Xstar);
rank2 = rank(X0_tensor10, 1e-6);

fprintf('Results:\n');
fprintf('  Final error: %.6e\n', error2);
fprintf('  Recovered rank: %d (true: %d)\n', rank2, rank_true);
fprintf('  Time: %.3f seconds\n\n', t2);

%% Test 3: Standard Power Method for Comparison
fprintf('--- Test 3: Standard Power Method (T_power=20) ---\n');
params3 = struct();
params3.r = rank_true;
params3.T_power = 20;
params3.Xstar = Xstar;
params3.projection = @(X) rank_projection(X, rank_true);

tic;
[X0_power, U0_power, history3] = initialize_power_method(y, operator, d, d, params3);
t3 = toc;

[error3, X0_power_aligned] = rectify_sign_ambiguity(X0_power, Xstar);
rank3 = rank(X0_power, 1e-6);

fprintf('Results:\n');
fprintf('  Final error: %.6e\n', error3);
fprintf('  Recovered rank: %d (true: %d)\n', rank3, rank_true);
fprintf('  Time: %.3f seconds\n\n', t3);

%% Comparison
fprintf('=== Comparison Summary ===\n');
fprintf('Method                  | Error        | Rank | Time     | Winner\n');
fprintf('---------------------------------------------------------------------\n');
fprintf('Tensor Lift (T=5)       | %.4e | %4d | %.3f s | %s\n', ...
    error1, rank1, t1, iif(error1 == min([error1, error2, error3]), '✓', ''));
fprintf('Tensor Lift (T=10)      | %.4e | %4d | %.3f s | %s\n', ...
    error2, rank2, t2, iif(error2 == min([error1, error2, error3]), '✓', ''));
fprintf('Power Method (T=20)     | %.4e | %4d | %.3f s | %s\n', ...
    error3, rank3, t3, iif(error3 == min([error1, error2, error3]), '✓', ''));

fprintf('\nBest initialization: ');
[best_error, best_idx] = min([error1, error2, error3]);
method_names = {'Tensor Lift (T=5)', 'Tensor Lift (T=10)', 'Power Method (T=20)'};
fprintf('%s (error: %.4e)\n', method_names{best_idx}, best_error);

%% Visualization
figure('Position', [100, 100, 1600, 1000]);

% Plot 1: Convergence - Tensor Lift T=5
subplot(2, 4, 1);
if isfield(history1, 'matrix_errors')
    semilogy(1:length(history1.matrix_errors), history1.matrix_errors, 'b-', 'LineWidth', 2);
    xlabel('Tensor Iteration');
    ylabel('Matrix Error');
    title('Tensor Lift (T=5) Convergence');
    grid on;
end

% Plot 2: Convergence - Tensor Lift T=10
subplot(2, 4, 2);
if isfield(history2, 'matrix_errors')
    semilogy(1:length(history2.matrix_errors), history2.matrix_errors, 'r-', 'LineWidth', 2);
    xlabel('Tensor Iteration');
    ylabel('Matrix Error');
    title('Tensor Lift (T=10) Convergence');
    grid on;
end

% Plot 3: Convergence - Power Method
subplot(2, 4, 3);
if isfield(history3, 'errors')
    semilogy(1:length(history3.errors), history3.errors, 'g-', 'LineWidth', 2);
    xlabel('Power Iteration');
    ylabel('Matrix Error');
    title('Power Method (T=20) Convergence');
    grid on;
end

% Plot 4: Comparison of Convergence
subplot(2, 4, 4);
hold on;
if isfield(history1, 'matrix_errors')
    semilogy(1:length(history1.matrix_errors), history1.matrix_errors, 'b-', 'LineWidth', 2, 'DisplayName', 'Tensor T=5');
end
if isfield(history2, 'matrix_errors')
    semilogy(1:length(history2.matrix_errors), history2.matrix_errors, 'r-', 'LineWidth', 2, 'DisplayName', 'Tensor T=10');
end
if isfield(history3, 'errors')
    semilogy(1:length(history3.errors), history3.errors, 'g-', 'LineWidth', 2, 'DisplayName', 'Power T=20');
end
xlabel('Iteration');
ylabel('Matrix Error');
title('Convergence Comparison');
legend('Location', 'best');
grid on;

% Plot 5: Ground Truth
subplot(2, 4, 5);
imagesc(Xstar);
colorbar;
title('Ground Truth');
axis equal tight;

% Plot 6: Tensor Lift T=5 Result
subplot(2, 4, 6);
imagesc(X0_tensor5_aligned);
colorbar;
title(sprintf('Tensor Lift T=5\nError: %.2e', error1));
axis equal tight;

% Plot 7: Tensor Lift T=10 Result
subplot(2, 4, 7);
imagesc(X0_tensor10_aligned);
colorbar;
title(sprintf('Tensor Lift T=10\nError: %.2e', error2));
axis equal tight;

% Plot 8: Power Method Result
subplot(2, 4, 8);
imagesc(X0_power_aligned);
colorbar;
title(sprintf('Power Method\nError: %.2e', error3));
axis equal tight;

%% Additional Analysis: Tensor vs Matrix Errors
if isfield(history1, 'tensor_errors') && isfield(history1, 'matrix_errors')
    figure('Position', [100, 100, 1200, 400]);
    
    subplot(1, 2, 1);
    semilogy(1:length(history1.tensor_errors), history1.tensor_errors, 'b-', 'LineWidth', 2, 'DisplayName', 'Tensor Error');
    hold on;
    semilogy(1:length(history1.matrix_errors), history1.matrix_errors, 'r-', 'LineWidth', 2, 'DisplayName', 'Matrix Error');
    xlabel('Iteration');
    ylabel('Relative Error');
    title('Tensor Lift (T=5): Tensor vs Matrix Errors');
    legend('Location', 'best');
    grid on;
    
    subplot(1, 2, 2);
    semilogy(1:length(history2.tensor_errors), history2.tensor_errors, 'b-', 'LineWidth', 2, 'DisplayName', 'Tensor Error');
    hold on;
    semilogy(1:length(history2.matrix_errors), history2.matrix_errors, 'r-', 'LineWidth', 2, 'DisplayName', 'Matrix Error');
    xlabel('Iteration');
    ylabel('Relative Error');
    title('Tensor Lift (T=10): Tensor vs Matrix Errors');
    legend('Location', 'best');
    grid on;
end

%% Save Results
fprintf('\n=== Saving Results ===\n');
results = struct();
results.Xstar = Xstar;
results.tensor5 = struct('X0', X0_tensor5_aligned, 'error', error1, 'rank', rank1, 'time', t1, 'history', history1);
results.tensor10 = struct('X0', X0_tensor10_aligned, 'error', error2, 'rank', rank2, 'time', t2, 'history', history2);
results.power = struct('X0', X0_power_aligned, 'error', error3, 'rank', rank3, 'time', t3, 'history', history3);
results.params = struct('d', d, 'm', m, 'rank_true', rank_true);

save('tensor_lift_initialization_results.mat', 'results');
fprintf('Results saved to: tensor_lift_initialization_results.mat\n');

fprintf('\n=== Test Complete ===\n');

%% Helper Functions
function X_proj = rank_projection(X, r)
    if r <= 0 || r >= min(size(X))
        X_proj = X;
        return;
    end
    [U, S, V] = svd(X, 'econ');
    r_effective = min(r, min(size(S)));
    S_proj = S;
    S_proj(r_effective+1:end, r_effective+1:end) = 0;
    X_proj = U * S_proj * V';
end

function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
