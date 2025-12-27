%% Test Tucker Tensor Spectral Initialization for Non-Symmetric Case
% Test spectral initialization for non-square, non-symmetric matrices
% Compares two local refinement methods:
%   1. Tucker RGD (Riemannian Gradient Descent on Tucker manifold)
%   2. PGD Amplitude (Projected Gradient Descent with amplitude loss)
%
% Test scenarios:
%   1. Non-square matrix: d1 ≠ d2
%   2. Non-symmetric Tucker: U1≠U3, U2≠U4 (extracted from HOSVD)
%   3. Compare convergence and final accuracy
%
% Test Date: 2025-01-XX

clear; clc;

fprintf('=== Test: Tucker Tensor Spectral Init (Non-Symmetric) ===\n\n');

%% Test Parameters
d1 = 20;             % First dimension (rows)
d2 = 20;             % Second dimension (cols)
r = 1;               % Tucker rank
m = 512;            % Number of measurements
T_power = 10;        % RGD iterations
mu = 0.1;           % Step size

rng(42);  % For reproducibility

fprintf('Test Configuration:\n');
fprintf('  Dimension d1 (rows): %d\n', d1);
fprintf('  Dimension d2 (cols): %d\n', d2);
fprintf('  Tucker rank r: %d\n', r);
fprintf('  Measurements m: %d\n', m);
fprintf('  RGD iterations: %d\n', T_power);
fprintf('  Step size: %.4f\n\n', mu);

%% Generate Ground Truth and Measurements
fprintf('=== Step 1: Generate Ground Truth and Measurements ===\n');

% Create ground truth matrix X_true (non-square, non-symmetric)
X_true = abs(randn(d1, r)) * abs(randn(r, d2));
X_true = X_true / norm(X_true, 'fro');
fprintf('Ground truth X_true: %dx%d (non-square, non-symmetric), norm=%.6f\n', ...
        d1, d2, norm(X_true, 'fro'));

% Generate random measurement matrices
fprintf('Generating %d random measurement matrices...\n', m);
A_cells = cell(m, 1);
A_matrix = zeros(m, d1*d2);
for i = 1:m
    Ai = randn(d1, d2);
    A_cells{i} = Ai;
    A_matrix(i, :) = Ai(:)';
end

% Compute measurements: y_i = |<Ai, X_true>|^2 / sqrt(m)
y = zeros(m, 1);
for i = 1:m
    y(i) = abs(sum(sum(A_cells{i} .* X_true)))^2 / sqrt(m);
end
fprintf('Measurements computed: y ∈ R^%d, norm=%.6f\n', m, norm(y));

% Create operator struct
operator = struct();
operator.A = @(X) forward_op(X, A_matrix, d1, d2);
operator.A_star = @(y_vec) reshape(A_matrix' * y_vec, [d1, d2]);
operator.A_cells = A_cells;

fprintf('\n');

%% Test 1: Method 1 - Tucker RGD (Spectral Init + Tucker manifold optimization)
fprintf('=== Test 1: Tucker RGD Refinement ===\n');
fprintf('Using spectral init + RGD on Tucker manifold (T_power=%d iterations)\n\n', T_power);

params_tucker = struct('T_power', T_power, 'mu', mu, 'r', r, ...
                       'Xstar', X_true, 'verbose', true, ...
                       'symmetric', false, ...  % Non-symmetric: use all HOSVD factors
                       'debug', false);  % Disable debug for speed

tic;
[X_tucker, ~, hist_tucker] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params_tucker);
time_tucker = toc;

[error_tucker, ~] = rectify_sign_ambiguity(X_tucker, X_true);

fprintf('\nResults (Tucker RGD):\n');
fprintf('  Final error: %.2e\n', error_tucker);
fprintf('  Total time: %.2fs\n', time_tucker);
fprintf('  Final loss: %.2e\n', hist_tucker.loss_function(end));
fprintf('  Iterations: %d\n', length(hist_tucker.loss_function));
fprintf('\n');

%% Test 2: Method 2 - PGD Amplitude (Spectral Init + PGD amplitude optimization)  
fprintf('=== Test 2: PGD Amplitude Refinement ===\n');
fprintf('Using spectral init + PGD amplitude (T_power=%d iterations)\n\n', T_power);

tic;  % Start timing for fair comparison

% Step 1: Get spectral initialization (T_power=0 to skip RGD iterations)
params_init_only = struct('T_power', 0, 'mu', mu, 'r', r, ...
                          'Xstar', X_true, 'verbose', false, ...
                          'symmetric', false, 'debug', false);

fprintf('Getting spectral initialization...\n');
[X_spectral_init, ~, ~] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params_init_only);
fprintf('Spectral init extracted from Tucker tensor\n');

% Step 2: Run PGD amplitude for T_power iterations
% Note: solve_PGD_amplitude expects projection function with 1 argument
% Use anonymous function to capture rank r
params_pgd = struct('T', T_power, ...
                    'mu', mu, ...
                    'r', r, ...
                    'Xstar', X_true, ...
                    'projection', @(X) project_low_rank(X, r));  % Capture r in closure

fprintf('Running PGD amplitude refinement...\n');
[output_pgd, X_pgd] = solve_PGD_amplitude(X_spectral_init, [], y, operator, d1, d2, r, m, params_pgd);

time_pgd = toc;  % Total time including spectral init

[error_pgd, ~] = rectify_sign_ambiguity(X_pgd, X_true);

fprintf('\nResults (PGD Amplitude):\n');
fprintf('  Final error: %.2e\n', error_pgd);
fprintf('  Total time: %.2fs\n', time_pgd);
fprintf('  Final loss: %.2e\n', output_pgd.Error_function(end));
fprintf('  Iterations: %d\n', params_pgd.T);
fprintf('\n');

%% Test 3: Direct TuckerTensor Spectral Initialization
fprintf('=== Test 3: Direct TuckerTensor Spectral Initialization ===\n');
fprintf('Testing TuckerTensor class spectral initialization directly\n\n');

% Non-symmetric case
dims = [d1, d2, d1, d2];
fprintf('Creating non-symmetric TuckerTensor...\n');
T_nonsym = TuckerTensor(dims, r, ...
                        'symmetric', false, ...
                        'init_method', 'spectral', ...
                        'operator', operator, ...
                        'y', y, ...  % Convert to tensor measurements
                        'm', m, ...
                        'debug', true);

fprintf('Non-symmetric TuckerTensor created:\n');
fprintf('  U1: %dx%d\n', size(T_nonsym.U{1}));
fprintf('  U2: %dx%d\n', size(T_nonsym.U{2}));
fprintf('  U3: %dx%d\n', size(T_nonsym.U{3}));
fprintf('  U4: %dx%d\n', size(T_nonsym.U{4}));
fprintf('  Core G: %s\n', mat2str(size(T_nonsym.G)));

% Check factor differences
diff_U13 = norm(T_nonsym.U{1} - T_nonsym.U{3}, 'fro');
diff_U24 = norm(T_nonsym.U{2} - T_nonsym.U{4}, 'fro');
fprintf('  ||U1 - U3||_F = %.6e\n', diff_U13);
fprintf('  ||U2 - U4||_F = %.6e\n', diff_U24);

if diff_U13 < 0.1 && diff_U24 < 0.1
    fprintf('  ✓ Partial symmetry detected: U1≈U3, U2≈U4\n');
elseif diff_U13 < 0.5 && diff_U24 < 0.5
    fprintf('  ~ Approximate partial symmetry (differences < 0.5)\n');
else
    fprintf('  ~ Weak or no partial symmetry detected\n');
end
fprintf('\n');

%% Test 4: Comparison Visualization
fprintf('=== Test 4: Comparison Visualization ===\n');

figure('Position', [100, 100, 1400, 900]);

% Loss convergence comparison
subplot(2, 3, 1);
semilogy(1:length(hist_tucker.loss_function), hist_tucker.loss_function, 'b-', 'LineWidth', 2);
hold on;
semilogy(1:length(output_pgd.Error_function), output_pgd.Error_function, 'r--', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Loss'); title('Loss Convergence'); 
legend('Tucker RGD', 'PGD Amplitude', 'Location', 'best');
grid on;

% Matrix error convergence comparison
subplot(2, 3, 2);
semilogy(1:length(hist_tucker.matrix_errors), hist_tucker.matrix_errors, 'b-', 'LineWidth', 2);
hold on;
semilogy(1:length(output_pgd.Error_Stand), output_pgd.Error_Stand, 'r--', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Matrix Error'); title('Matrix Error vs Ground Truth'); 
legend('Tucker RGD', 'PGD Amplitude', 'Location', 'best');
grid on;

% Final error comparison
subplot(2, 3, 3);
bar([error_tucker, error_pgd]);
set(gca, 'XTickLabel', {'Tucker RGD', 'PGD Amplitude'});
ylabel('Final Error'); title('Final Error Comparison'); 
grid on;
set(gca, 'YScale', 'log');

% Time comparison
subplot(2, 3, 4);
bar([time_tucker, time_pgd]);
set(gca, 'XTickLabel', {'Tucker RGD', 'PGD Amplitude'});
ylabel('Time (seconds)'); title('Computation Time'); 
grid on;

% Convergence rate (log scale)
subplot(2, 3, 5);
% Compute convergence rate: log(error(t))
if length(hist_tucker.matrix_errors) > 10
    conv_rate_tucker = diff(log10(hist_tucker.matrix_errors + 1e-16));
    plot(1:length(conv_rate_tucker), conv_rate_tucker, 'b-', 'LineWidth', 2);
    hold on;
end
if length(output_pgd.Error_Stand) > 10
    conv_rate_pgd = diff(log10(output_pgd.Error_Stand + 1e-16));
    plot(1:length(conv_rate_pgd), conv_rate_pgd, 'r--', 'LineWidth', 2);
end
xlabel('Iteration'); ylabel('log₁₀(Error) Change'); 
title('Convergence Rate (per iteration)');
legend('Tucker RGD', 'PGD Amplitude', 'Location', 'best');
grid on;

% Factor matrix differences (for Tucker method)
subplot(2, 3, 6);
bar([diff_U13, diff_U24]);
set(gca, 'XTickLabel', {'||U1-U3||', '||U2-U4||'});
ylabel('Frobenius Norm'); title('Factor Matrix Differences (Tucker)'); 
grid on;

sgtitle(sprintf('Tucker RGD vs PGD Amplitude (d1=%d, d2=%d, r=%d, m=%d)', d1, d2, r, m));

fprintf('Comparison plots generated.\n\n');

%% Summary
fprintf('=== Summary: Local Refinement Comparison ===\n');
fprintf('\nMethod 1: Tucker RGD (Spectral Init + Tucker manifold optimization)\n');
fprintf('  - Optimization on Tucker tensor manifold\n');
fprintf('  - Uses retraction to maintain Tucker structure\n');
fprintf('  - Non-symmetric: U1, U2, U3, U4 may differ\n');
fprintf('  - Final error: %.6e\n', error_tucker);
fprintf('  - Final loss:  %.6e\n', hist_tucker.loss_function(end));
fprintf('  - Time: %.2fs\n', time_tucker);
fprintf('  - Iterations: %d\n', length(hist_tucker.loss_function));

fprintf('\nMethod 2: PGD Amplitude (Spectral Init + PGD amplitude optimization)\n');
fprintf('  - Projected Gradient Descent with amplitude loss\n');
fprintf('  - Projection onto low-rank manifold\n');
fprintf('  - Direct matrix optimization (not tensor-based)\n');
fprintf('  - Final error: %.6e\n', error_pgd);
fprintf('  - Final loss:  %.6e\n', output_pgd.Error_function(end));
fprintf('  - Time: %.2fs\n', time_pgd);
fprintf('  - Iterations: %d\n', params_pgd.T);

fprintf('\nComparison:\n');
error_ratio = error_tucker / error_pgd;
time_ratio = time_tucker / time_pgd;
fprintf('  Error ratio (Tucker/PGD): %.3f\n', error_ratio);
fprintf('  Time ratio (Tucker/PGD):  %.3f\n', time_ratio);

if error_tucker < error_pgd * 0.95
    fprintf('  ✓✓ Tucker RGD is MORE ACCURATE (%.1f%% better)\n', ...
            (1 - error_tucker/error_pgd) * 100);
elseif error_pgd < error_tucker * 0.95
    fprintf('  ✓✓ PGD Amplitude is MORE ACCURATE (%.1f%% better)\n', ...
            (1 - error_pgd/error_tucker) * 100);
else
    fprintf('  ≈ Both methods achieve similar accuracy\n');
end

if time_tucker < time_pgd * 0.95
    fprintf('  ⚡ Tucker RGD is FASTER (%.2fx speedup)\n', time_pgd/time_tucker);
elseif time_pgd < time_tucker * 0.95
    fprintf('  ⚡ PGD Amplitude is FASTER (%.2fx speedup)\n', time_tucker/time_pgd);
else
    fprintf('  ≈ Both methods have similar computation time\n');
end

fprintf('\nFactor Matrix Structure (Tucker method):\n');
fprintf('  ||U1 - U3||_F = %.6e\n', diff_U13);
fprintf('  ||U2 - U4||_F = %.6e\n', diff_U24);
if diff_U13 < 0.1 && diff_U24 < 0.1
    fprintf('  ✓ Strong partial symmetry: U1≈U3, U2≈U4\n');
    fprintf('    (Expected for spectral tensor H = sum_i y_i * (Ai ⊗ Ai))\n');
elseif diff_U13 < 0.5 && diff_U24 < 0.5
    fprintf('  ~ Approximate partial symmetry detected\n');
else
    fprintf('  ~ Weak partial symmetry (may indicate finite m or noise)\n');
end

fprintf('\nRecommendation:\n');
if error_tucker < error_pgd && time_tucker < time_pgd * 1.5
    fprintf('  ✓✓ Use Tucker RGD: Better accuracy, reasonable speed\n');
elseif error_pgd < error_tucker && time_pgd < time_tucker * 1.5
    fprintf('  ✓✓ Use PGD Amplitude: Better accuracy, reasonable speed\n');
elseif error_tucker < error_pgd
    fprintf('  ✓ Use Tucker RGD: Better accuracy (at %.1fx time cost)\n', time_ratio);
elseif time_pgd < time_tucker
    fprintf('  ✓ Use PGD Amplitude: Faster (with %.2f%% accuracy tradeoff)\n', ...
            (1 - error_pgd/error_tucker) * 100);
else
    fprintf('  ≈ Both methods are comparable, choose based on preference\n');
end

fprintf('\n✓ Test Complete\n');

%% Helper Functions

function y = forward_op(X, A_matrix, d1, d2)
    % Forward operator: y_i = |<Ai, X>|^2 / sqrt(m)
    % For phase retrieval measurements
    m = size(A_matrix, 1);
    y = zeros(m, 1);
    for i = 1:m
        Ai = reshape(A_matrix(i, :), [d1, d2]);
        y(i) = abs(sum(sum(Ai .* X)))^2 / sqrt(m);
    end
end

function X_proj = project_low_rank(X, r)
    % Project matrix X onto rank-r manifold via SVD truncation
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

