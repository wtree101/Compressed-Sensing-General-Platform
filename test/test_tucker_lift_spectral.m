%% Test initialize_tensor_lift_tucker with Spectral Initialization
% Test the updated initialize_tensor_lift_tucker function with the new
% spectral initialization method
%
% This test compares:
%   1. Spectral initialization (no RGD iterations)
%   2. Spectral initialization + RGD refinement

clear; clc;

fprintf('=== Test initialize_tensor_lift_tucker with Spectral Init ===\n\n');

%% Setup
d = 100; r = 1; m = 1000;
use_preprocessing = true;  % Test with preprocessing
rng(42);

% Ground truth
U_true = randn(d, r);
Xstar = U_true * U_true';
Xstar = Xstar / norm(Xstar, 'fro');

% Create measurement operator
A = randn(m, d*d);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d, d]);

% Generate measurements: phase retrieval (magnitude only)
y = abs(operator.A(Xstar)) / sqrt(m);

fprintf('Setup: d=%d, r=%d, m=%d\n', d, r, m);
fprintf('Ground truth: rank-%d matrix\n\n', r);

%% Test 1: Spectral Initialization Only (T_power = 0)
fprintf('=== Test 1: Spectral Initialization Only ===\n');

params1 = struct();
params1.T_power = 1;  % No RGD iterations, just spectral init
params1.r = r;
params1.Xstar = Xstar;
params1.verbose = true;  % verbose=true automatically enables debug mode
params1.symmetric = true;
params1.init_method = 'spectral';

if use_preprocessing
    params1.pre_func = @(y) set_zero_outside_range_tensor(y);
end

tic;
[X_spectral, ~, hist1] = initialize_tensor_lift_tucker_spectral(y, operator, d, d, params1);
time1 = toc;

[error1, ~] = rectify_sign_ambiguity(X_spectral, Xstar);

fprintf('Spectral init only:\n');
fprintf('  Error: %.6e\n', error1);
fprintf('  Time:  %.2f seconds\n\n', time1);

%% Test 2: Spectral Init + RGD Refinement
fprintf('=== Test 2: Spectral Init + RGD Refinement ===\n');

params2 = struct();
params2.T_power = 50;  % 50 RGD iterations
params2.mu = 0.1;
params2.r = r;
params2.Xstar = Xstar;
params2.verbose = true;  % verbose=true automatically enables debug mode
params2.symmetric = true;
params2.init_method = 'spectral';

if use_preprocessing
    params2.pre_func = @(y) set_zero_outside_range_tensor(y);
end

tic;
[X_rgd, ~, hist2] = initialize_tensor_lift_tucker_spectral(y, operator, d, d, params2);
time2 = toc;

[error2, ~] = rectify_sign_ambiguity(X_rgd, Xstar);

fprintf('Spectral init + RGD:\n');
fprintf('  Error: %.6e (after %d iterations)\n', error2, params2.T_power);
fprintf('  Time:  %.2f seconds\n\n', time2);

%% Test 3: Comparison with Traditional Init + RGD
fprintf('=== Test 3: Traditional Ones Init + RGD ===\n');

params3 = struct();
params3.T_power = 50;
params3.mu = 0.1;
params3.r = r;
params3.Xstar = Xstar;
params3.verbose = true;  % verbose=true automatically enables debug mode
params3.symmetric = true;
params3.init_method = 'ones';  % Traditional initialization

tic;
[X_ones, ~, hist3] = initialize_tensor_lift_tucker_spectral(y, operator, d, d, params3);
time3 = toc;

[error3, ~] = rectify_sign_ambiguity(X_ones, Xstar);

fprintf('Ones init + RGD:\n');
fprintf('  Error: %.6e (after %d iterations)\n', error3, params3.T_power);
fprintf('  Time:  %.2f seconds\n\n', time3);

%% Results Summary
fprintf('=== Results Summary ===\n');
fprintf('Method                        | Error      | Time (s) | Improvement\n');
fprintf('------------------------------|------------|----------|------------\n');
fprintf('1. Spectral only              | %.4e | %7.2f  | baseline\n', error1, time1);
fprintf('2. Spectral + RGD (%2d iter)  | %.4e | %7.2f  | %.1f%%\n', ...
        params2.T_power, error2, time2, (error1-error2)/error1*100);
fprintf('3. Ones + RGD (%2d iter)      | %.4e | %7.2f  | %.1f%%\n', ...
        params3.T_power, error3, time3, (error1-error3)/error1*100);
fprintf('\n');

fprintf('Spectral vs Ones comparison:\n');
if error2 < error3
    fprintf('  Spectral + RGD is %.1fx better than Ones + RGD\n', error3/error2);
else
    fprintf('  Ones + RGD is %.1fx better than Spectral + RGD\n', error2/error3);
end
fprintf('\n');

%% Convergence Visualization
fprintf('=== Generating Convergence Plots ===\n');

figure('Position', [100, 100, 1400, 500]);

% Plot 1: Loss convergence comparison
subplot(1, 3, 1);
if params2.T_power > 0
    semilogy(hist2.loss_function, 'b-', 'LineWidth', 2); hold on;
end
if params3.T_power > 0
    semilogy(hist3.loss_function, 'r--', 'LineWidth', 2);
end
xlabel('Iteration'); ylabel('Loss');
title('Loss Convergence');
legend('Spectral + RGD', 'Ones + RGD', 'Location', 'best');
grid on;

% Plot 2: Matrix error convergence
subplot(1, 3, 2);
if params2.T_power > 0
    semilogy(hist2.matrix_errors, 'b-', 'LineWidth', 2); hold on;
end
if params3.T_power > 0
    semilogy(hist3.matrix_errors, 'r--', 'LineWidth', 2);
end
yline(error1, 'g:', 'Spectral Init', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Matrix Error');
title('Error Convergence');
legend('Spectral + RGD', 'Ones + RGD', 'Spectral Init', 'Location', 'best');
grid on;

% Plot 3: Final error comparison
subplot(1, 3, 3);
errors = [error1, error2, error3];
bar(errors);
set(gca, 'XTickLabel', {'Spectral', 'Spectral+RGD', 'Ones+RGD'});
ylabel('Relative Error');
title('Final Error Comparison');
grid on;
set(gca, 'YScale', 'log');

if use_preprocessing
    preproc_str = 'WITH preprocessing';
else
    preproc_str = 'NO preprocessing';
end
sgtitle(sprintf('initialize\\_tensor\\_lift\\_tucker: Spectral Init (d=%d, r=%d, m=%d) - %s', ...
        d, r, m, preproc_str));

fprintf('Plots generated.\n\n');

fprintf('✓ Test Complete\n');

