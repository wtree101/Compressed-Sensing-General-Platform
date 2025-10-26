%%%%%%%%%% Test PGD with Amplitude-Based Loss
% Test the new solve_PGD_amplitude solver for low-rank phase retrieval

clear; clc; close all;

fprintf('=== PGD Amplitude Loss Test ===\n\n');

%% Problem Setup
d1 = 20; d2 = 20;
n = d1 * d2;
m = 800;
rank_true = 2;
T = 300;

fprintf('Configuration: %dx%d matrix, rank=%d, m=%d measurements\n', d1, d2, rank_true, m);

%% Generate Low-Rank Ground Truth
U_true = randn(d1, rank_true);
V_true = randn(d2, rank_true);
Xstar = U_true * V_true';
Xstar = Xstar / norm(Xstar, 'fro');

fprintf('Ground truth rank: %d\n', rank(Xstar));
fprintf('Ground truth norm: %.6f\n', norm(Xstar, 'fro'));

%% Generate Measurements
A = randn(m, n) ;
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(z) reshape(A' * z, [d1, d2]);

% Phase retrieval: observe only magnitudes
y = abs(operator.A(Xstar))/sqrt(m);

fprintf('Measurement range: [%.3f, %.3f]\n\n', min(y), max(y));

%% Initialize with Power Method
fprintf('--- Spectral Initialization ---\n');
params_init = struct();
params_init.is_matrix = true;
params_init.r = rank_true;
params_init.projection = @(X, p) project_low_rank(X, p.r);

T_power = 30;
[Xl_init, ~] = initialize_power_method(y, operator, d1, d2, T_power, params_init);

[init_error, Xl_init] = rectify_sign_ambiguity(Xl_init, Xstar);
fprintf('Initialization error (rectified): %.6e\n\n', init_error);

%% Setup PGD Parameters
params = struct();
params.T = T;
params.Xstar = Xstar;
params.mu = 0.3;  % Step size for amplitude loss
params.r = rank_true;
params.projection = @(X, p) project_low_rank(X, p.r);

%% Run PGD with Amplitude Loss
fprintf('--- Running PGD-Amplitude ---\n');
tic;
[solver_output, Xl_final] = solve_AP(Xl_init, [], y, operator, d1, d2, [], m, params);
Error_Stand = solver_output.Error_Stand;
Error_function = solver_output.Error_function;
elapsed_time = toc;

fprintf('\nResults:\n');
fprintf('  Final relative error: %.6e\n', Error_Stand(end));
fprintf('  Final amplitude loss: %.6e\n', Error_function(end));
fprintf('  Recovered rank: %d (true: %d)\n', rank(Xl_final, 1e-6), rank_true);
fprintf('  Computation time: %.3f seconds\n', elapsed_time);

%% Check Recovery Quality
[recovery_error, Xl_final_aligned] = rectify_sign_ambiguity(Xl_final, Xstar);
[~, S_true, ~] = svd(Xstar);
[~, S_recovered, ~] = svd(Xl_final);

fprintf('\nDetailed Analysis:\n');
fprintf('  True singular values: [');
fprintf('%.3f ', diag(S_true(1:min(5, rank_true), 1:min(5, rank_true))));
fprintf(']\n');
fprintf('  Recovered singular values: [');
fprintf('%.3f ', diag(S_recovered(1:min(5, d1), 1:min(5, d2))));
fprintf(']\n');

if recovery_error < 1e-3
    fprintf('  Status: RECOVERED ✓\n');
else
    fprintf('  Status: Partial recovery\n');
end

%% Visualization
figure('Position', [100, 100, 1400, 900]);

% Plot 1: Convergence - Relative Error
subplot(2, 3, 1);
semilogy(1:T, Error_Stand, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Relative Error');
title('Convergence: ||X - X*||_F / ||X*||_F');
grid on;

% Plot 2: Amplitude Loss
subplot(2, 3, 2);
semilogy(1:T, Error_function, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Amplitude Loss');
title('Amplitude Loss: (1/2m)||y - |A(X)|||^2');
grid on;

% Plot 3: Log-scale convergence
subplot(2, 3, 3);
semilogy(1:T, Error_Stand, 'b-', 'LineWidth', 2);
hold on;
semilogy(1:T, Error_function, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Error (log scale)');
title('Combined Convergence');
legend('Relative Error', 'Amplitude Loss', 'Location', 'best');
grid on;

% Plot 4: Ground Truth
subplot(2, 3, 4);
imagesc(Xstar);
colorbar;
title('Ground Truth Matrix');
axis equal tight;

% Plot 5: Recovered Matrix
subplot(2, 3, 5);
imagesc(Xl_final);
colorbar;
title(sprintf('Recovered Matrix (Error: %.2e)', recovery_error));
axis equal tight;

% Plot 6: Recovery Error (sign-aligned)
subplot(2, 3, 6);
imagesc(Xl_final_aligned - Xstar);
colorbar;
title('Recovery Error: X_{final} - X* (aligned)');
axis equal tight;

fprintf('\n=== Test Complete ===\n');

%% Helper Function
function X_proj = project_low_rank(X, r)
    if r <= 0 || r >= min(size(X))
        X_proj = X;
        return;
    end
    [U, S, V] = svd(X, 'econ');
    S_proj = S;
    if size(S, 1) > r && size(S, 2) > r
        S_proj(r+1:end, r+1:end) = 0;
    end
    X_proj = U * S_proj * V';
end
