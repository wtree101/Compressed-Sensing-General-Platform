%%%%%%%%%% Comparison Test: AP vs PGD-Amplitude
% Compare AP and PGD-Amplitude algorithms for low-rank phase retrieval

clear; clc; close all;

fprintf('=== AP vs PGD-Amplitude Comparison Test ===\n\n');

%% Problem Setup
d1 = 20; d2 = 20;
n = d1 * d2;
m = 400;
rank_true = 1;
T = 100;

fprintf('Configuration: %dx%d matrix, rank=%d, m=%d measurements\n', d1, d2, rank_true, m);
fprintf('Iterations: %d\n\n', T);

%% Generate Low-Rank Ground Truth
U_true = randn(d1, rank_true);
V_true = randn(d2, rank_true);


Xstar = U_true * V_true';
Xstar = Xstar / norm(Xstar, 'fro');

% Can we shift the ground truth to be all positive, e.g., by adding a rank-1 all-ones matrix?



fprintf('Ground truth rank: %d\n', rank(Xstar));
fprintf('Ground truth norm: %.6f\n', norm(Xstar, 'fro'));

%% Generate Measurements
A = randn(m, n);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(z) reshape(A' * z, [d1, d2]);

% Phase retrieval: observe only magnitudes
AX_abs = abs(operator.A(Xstar))/sqrt(m);

Z = ones([d1,d2])*10;
AZ = operator.A(Z)/sqrt(m);
% approximate |A(X_star+Z)|
% Solve x from x^2 - 2*a*x + a^2 = y^2 element-wise, positive root
% Inputs: a, y are vectors of same length
% Output: x is the positive root for each element



% % find approximate |A(Xstar + Z)| using positive root
 y = AX_abs;


fprintf('Measurement range: [%.3f, %.3f]\n\n', min(y), max(y));

%% Initialize with Power Method
fprintf('--- Spectral Initialization (Power Method) ---\n');
params_init = struct();
params_init.r = rank_true;
params_init.T_power = 30;
params_init.projection = @(X) project_low_rank(X, rank_true);
params_init.Xstar = Xstar;  % For error tracking

[Xl_init_raw, ~, init_history] = initialize_power_method(y, operator, d1, d2, params_init);

% Rectify sign ambiguity for initialization
[init_error, Xl_init] = rectify_sign_ambiguity(Xl_init_raw, Xstar);
fprintf('Initialization error (rectified): %.6e\n', init_error);
fprintf('Initialization rank: %d\n\n', rank(Xl_init, 1e-6));

%% Setup Common Parameters
params_common = struct();
params_common.T = T;
params_common.Xstar = Xstar;
params_common.r = rank_true;
params_common.projection = @(X) project_low_rank(X, rank_true);
params_common.nonlinear_func = @(z) abs(z);
params_common.lambda = 0;

%% Test 1: Alternating Projection (AP)
fprintf('--- Running Alternating Projection (AP) ---\n');
params_AP = params_common;
params_AP.mu = 0.01;  % Step size for PGD subproblem in AP

tic;
[solver_output_AP, Xl_final_AP] = solve_AP(Xl_init, [], y, operator, d1, d2, [], [], params_AP);
Error_Stand_AP = solver_output_AP.Error_Stand;
Error_function_AP = solver_output_AP.Error_function;
time_AP = toc;

[final_error_AP, Xl_aligned_AP] = rectify_sign_ambiguity(Xl_final_AP, Xstar);
rank_AP = rank(Xl_final_AP, 1e-6);

fprintf('AP Results:\n');
fprintf('  Final relative error: %.6e\n', final_error_AP);
fprintf('  Final function error: %.6e\n', Error_function_AP(end));
fprintf('  Recovered rank: %d (true: %d)\n', rank_AP, rank_true);
fprintf('  Computation time: %.3f seconds\n', time_AP);
fprintf('  Status: %s\n\n', iif(final_error_AP < 1e-3, 'RECOVERED ✓', 'Partial recovery'));

%% Test 2: PGD with Amplitude Loss
fprintf('--- Running PGD-Amplitude ---\n');
params_PGD = params_common;
params_PGD.mu = 0.3;  % Step size for amplitude-based gradient

tic;
[solver_output_PGD, Xl_final_PGD] = solve_PGD_amplitude(Xl_init, [], y, operator, d1, d2, [], m, params_PGD);
Error_Stand_PGD = solver_output_PGD.Error_Stand;
Error_function_PGD = solver_output_PGD.Error_function;
time_PGD = toc;

[final_error_PGD, Xl_aligned_PGD] = rectify_sign_ambiguity(Xl_final_PGD, Xstar);
rank_PGD = rank(Xl_final_PGD, 1e-6);

fprintf('PGD-Amplitude Results:\n');
fprintf('  Final relative error: %.6e\n', final_error_PGD);
fprintf('  Final amplitude loss: %.6e\n', Error_function_PGD(end));
fprintf('  Recovered rank: %d (true: %d)\n', rank_PGD, rank_true);
fprintf('  Computation time: %.3f seconds\n', time_PGD);
fprintf('  Status: %s\n\n', iif(final_error_PGD < 1e-3, 'RECOVERED ✓', 'Partial recovery'));

%% Comparison Summary
fprintf('=== Comparison Summary ===\n');
fprintf('Metric                    AP              PGD-Amplitude   Winner\n');
fprintf('---------------------------------------------------------------------\n');
fprintf('Final Rel. Error:       %.4e      %.4e      %s\n', ...
    final_error_AP, final_error_PGD, iif(final_error_AP < final_error_PGD, 'AP', 'PGD'));
fprintf('Computation Time:       %.3f sec       %.3f sec       %s\n', ...
    time_AP, time_PGD, iif(time_AP < time_PGD, 'AP', 'PGD'));
fprintf('Recovered Rank:         %d               %d               %s\n', ...
    rank_AP, rank_PGD, iif(rank_AP == rank_true, 'AP', iif(rank_PGD == rank_true, 'PGD', 'Tie')));
fprintf('Speedup factor:         %.2fx (AP is %s)\n', ...
    max(time_AP, time_PGD) / min(time_AP, time_PGD), ...
    iif(time_AP < time_PGD, 'faster', 'slower'));
fprintf('Error improvement:      %.2fx (AP is %s)\n', ...
    max(final_error_AP, final_error_PGD) / min(final_error_AP, final_error_PGD), ...
    iif(final_error_AP < final_error_PGD, 'better', 'worse'));

%% Detailed Analysis
fprintf('\n=== Singular Values Comparison ===\n');
[~, S_true, ~] = svd(Xstar);
[~, S_AP, ~] = svd(Xl_final_AP);
[~, S_PGD, ~] = svd(Xl_final_PGD);

fprintf('True:         [');
fprintf('%.4f ', diag(S_true(1:min(5, rank_true), 1:min(5, rank_true))));
fprintf(']\n');
fprintf('AP:           [');
fprintf('%.4f ', diag(S_AP(1:min(5, d1), 1:min(5, d2))));
fprintf(']\n');
fprintf('PGD-Amp:      [');
fprintf('%.4f ', diag(S_PGD(1:min(5, d1), 1:min(5, d2))));
fprintf(']\n');

%% Visualization
figure('Position', [100, 100, 1600, 1000]);

% Plot 1: Convergence Comparison - Relative Error
subplot(2, 4, 1);
semilogy(1:T, Error_Stand_AP, 'b-', 'LineWidth', 2, 'DisplayName', 'AP');
hold on;
semilogy(1:T, Error_Stand_PGD, 'r-', 'LineWidth', 2, 'DisplayName', 'PGD-Amplitude');
xlabel('Iteration');
ylabel('Relative Error');
title('Convergence: Relative Error');
legend('Location', 'best');
grid on;

% Plot 2: Function Error Comparison
subplot(2, 4, 2);
semilogy(1:T, Error_function_AP, 'b-', 'LineWidth', 2, 'DisplayName', 'AP');
hold on;
semilogy(1:T, Error_function_PGD, 'r-', 'LineWidth', 2, 'DisplayName', 'PGD-Amplitude');
xlabel('Iteration');
ylabel('Function Error');
title('Function Error');
legend('Location', 'best');
grid on;

% Plot 3: Error Ratio
subplot(2, 4, 3);
error_ratio = Error_Stand_AP ./ (Error_Stand_PGD + 1e-15);
plot(1:T, error_ratio, 'k-', 'LineWidth', 2);
hold on;
plot([1 T], [1 1], 'k--', 'LineWidth', 1);
xlabel('Iteration');
ylabel('Error Ratio (AP/PGD)');
title('AP vs PGD Error Ratio');
grid on;
ylim([0, max(5, max(error_ratio))]);

% Plot 4: Convergence Rate (log scale)
subplot(2, 4, 4);
if T > 50
    % Estimate linear convergence rate
    start_idx = max(1, T - 100);
    p_AP = polyfit(start_idx:T, log10(Error_Stand_AP(start_idx:T)), 1);
    p_PGD = polyfit(start_idx:T, log10(Error_Stand_PGD(start_idx:T)), 1);
    
    semilogy(1:T, Error_Stand_AP, 'b-', 'LineWidth', 2);
    hold on;
    semilogy(1:T, Error_Stand_PGD, 'r-', 'LineWidth', 2);
    
    % Plot fitted lines
    semilogy(start_idx:T, 10.^polyval(p_AP, start_idx:T), 'b--', 'LineWidth', 1);
    semilogy(start_idx:T, 10.^polyval(p_PGD, start_idx:T), 'r--', 'LineWidth', 1);
    
    title(sprintf('Convergence Rates\nAP: %.3f, PGD: %.3f', p_AP(1), p_PGD(1)));
    xlabel('Iteration');
    ylabel('Error (log)');
    grid on;
end

% Plot 5: Ground Truth
subplot(2, 4, 5);
imagesc(Xstar);
colorbar;
title('Ground Truth');
axis equal tight;

% Plot 6: AP Result
subplot(2, 4, 6);
imagesc(Xl_aligned_AP);
colorbar;
title(sprintf('AP Result\nError: %.2e', final_error_AP));
axis equal tight;

% Plot 7: PGD Result
subplot(2, 4, 7);
imagesc(Xl_aligned_PGD);
colorbar;
title(sprintf('PGD-Amplitude Result\nError: %.2e', final_error_PGD));
axis equal tight;

% Plot 8: Error Comparison
subplot(2, 4, 8);
imagesc([Xl_aligned_AP - Xstar, Xl_aligned_PGD - Xstar]);
colorbar;
title('Recovery Errors [AP | PGD]');
axis equal tight;

%% Save Results
fprintf('\n=== Saving Results ===\n');
save('ap_vs_pgd_amplitude_comparison.mat', 'Error_Stand_AP', 'Error_Stand_PGD', ...
    'Error_function_AP', 'Error_function_PGD', 'Xl_final_AP', 'Xl_final_PGD', ...
    'Xstar', 'time_AP', 'time_PGD', 'params_AP', 'params_PGD');
fprintf('Results saved to: ap_vs_pgd_amplitude_comparison.mat\n');

fprintf('\n=== Test Complete ===\n');

%% Helper Functions
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

function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

function x = solve_positive_root(a, y)
    assert(all(y >= 0) && all(a >= 0), 'Inputs y and a must be non-negative');
    % x^2 - 2*a*x + a^2 - y^2 = 0
    % => (x - a)^2 = y^2
    % => x - a = y or x - a = -y
    % Positive root: x = a + abs(y)
    x = a + abs(y);
end