%%%%%%%%%% Simple AP Phase Retrieval Test
% Quick test for AP algorithm on sparse phase retrieval

clear; clc; close all;

fprintf('=== Simple AP Phase Retrieval Test ===\n');

%% Simple Problem Setup
d1 = 20; d2 = 1;        % Vector signal (1D)
n = d1 * d2;
m = 60;                 % Number of measurements
sparsity = 5;           % Sparsity level
T = 200;                % Iterations

%% Generate Sparse Signal
support_idx = randperm(n, sparsity);
x_true = zeros(n, 1);
x_true(support_idx) = randn(sparsity, 1) + 1i * randn(sparsity, 1);
Xstar = reshape(x_true, [d1, d2]);

%% Generate Measurements
A = (randn(m, n) + 1i * randn(m, n)) / sqrt(2 * m);
operator.A = @(X) A * X(:);
operator.A_star = @(z) reshape(A' * z, [d1, d2]);

z_true = operator.A(Xstar);
y = abs(z_true); % Magnitude measurements

%% Setup AP Parameters
params = struct();
params.T = T;
params.Xstar = Xstar;
params.nonlinear_func = @(z) abs(z);
params.sparsity = sparsity;
params.projection = @(X, p) project_sparse(X, p.sparsity);
params.mu = 0.01;
params.lambda = 0;

%% Initialize and Run AP
Xl_init = randn(d1, d2) + 1i * randn(d1, d2);

fprintf('Running AP algorithm...\n');
tic;
[Error_Stand, Error_function, Xl_final] = solve_AP(Xl_init, [], y, operator, d1, d2, [], [], params);
elapsed_time = toc;

%% Analyze Results
final_error = Error_Stand(end);
recovered_sparsity = nnz(abs(Xl_final(:)) > 0.1 * max(abs(Xl_final(:))));
true_support = find(abs(Xstar(:)) > 1e-10);
recovered_support = find(abs(Xl_final(:)) > 0.1 * max(abs(Xl_final(:))));
support_recovery = length(intersect(true_support, recovered_support));

fprintf('\n=== Results ===\n');
fprintf('Final relative error: %.6f\n', final_error);
fprintf('True sparsity: %d\n', sparsity);
fprintf('Recovered sparsity: %d\n', recovered_sparsity);
fprintf('Support recovery: %d/%d (%.1f%%)\n', support_recovery, sparsity, 100*support_recovery/sparsity);
fprintf('Computation time: %.3f seconds\n', elapsed_time);

% Check success
if final_error < 0.1 && support_recovery >= 0.8 * sparsity
    fprintf('Status: SUCCESS ✓\n');
else
    fprintf('Status: FAILED\n');
end

%% Plot Results
figure('Position', [100, 100, 1200, 800]);

% Plot convergence
subplot(2, 3, 1);
semilogy(1:T, Error_Stand, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Relative Error');
title('AP Convergence');
grid on;

subplot(2, 3, 2);
semilogy(1:T, Error_function, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Function Error');
title('Function Error');
grid on;

% Plot signals
subplot(2, 3, 3);
stem(1:n, abs(Xstar(:)), 'b', 'LineWidth', 2);
title('True Signal (Magnitude)');
xlabel('Index');
ylabel('Magnitude');

subplot(2, 3, 4);
stem(1:n, abs(Xl_final(:)), 'r', 'LineWidth', 2);
title('Recovered Signal (Magnitude)');
xlabel('Index');
ylabel('Magnitude');

% Support comparison
subplot(2, 3, 5);
hold on;
stem(true_support, ones(length(true_support), 1), 'bo', 'MarkerSize', 8, 'DisplayName', 'True');
stem(recovered_support, 0.5*ones(length(recovered_support), 1), 'ro', 'MarkerSize', 8, 'DisplayName', 'Recovered');
xlabel('Index');
ylabel('Support');
title('Support Comparison');
legend('show');
ylim([0, 1.2]);

% Error visualization
subplot(2, 3, 6);
stem(1:n, abs(Xl_final(:) - Xstar(:)), 'g', 'LineWidth', 2);
title('Recovery Error');
xlabel('Index');
ylabel('Error Magnitude');

fprintf('\nTest completed. Check plots for visual verification.\n');

%% Helper Function
function X_proj = project_sparse(X, s)
    X_vec = X(:);
    [~, idx] = sort(abs(X_vec), 'descend');
    X_proj_vec = zeros(size(X_vec));
    X_proj_vec(idx(1:s)) = X_vec(idx(1:s));
    X_proj = reshape(X_proj_vec, size(X));
end
