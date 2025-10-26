%%%%%%%%%% Simple Test for Power Method Initialization - Sparse Phase Retrieval
% Test power method initialization for sparse phase retrieval

clear; clc; close all;

fprintf('=== Power Method Initialization Test (Sparse) ===\n\n');

%% Problem Setup
n = 100;                % Signal dimension
d1 = n; d2 = 1;         % Vector dimensions
m = 20;                % Number of measurements
sparsity = 5;           % Number of non-zero elements
T_power = 100;          % Power iterations

fprintf('Configuration: n=%d, sparsity=%d, m=%d measurements\n', n, sparsity, m);

%% Generate Sparse Ground Truth Signal
support_idx = randperm(n, sparsity);
x_true = zeros(n, 1);
x_true(support_idx) = randn(sparsity, 1);
Xstar = x_true / norm(x_true);

fprintf('True sparsity: %d, locations: [%s]\n', sparsity, num2str(support_idx(1:min(5,end))));

%% Generate Measurements
A = randn(m, n) / sqrt(m);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(z) reshape(A' * z, [d1, d2]);

y = abs(operator.A(Xstar));

%% Test 1: Power Method WITHOUT Projection
fprintf('\n--- Test 1: Without Projection ---\n');
params1 = struct();
params1.is_matrix = false;

tic;
[Xl_init1, ~] = initialize_power_method(y, operator, d1, d2, T_power, params1);
t1 = toc;

error1 = norm(Xl_init1 - Xstar, 'fro') / norm(Xstar, 'fro');
sparsity1 = nnz(abs(Xl_init1) > 0.1 * max(abs(Xl_init1)));
fprintf('  Time: %.3f sec\n', t1);
fprintf('  Relative error: %.4e\n', error1);
fprintf('  Recovered sparsity: %d (true: %d)\n', sparsity1, sparsity);

%% Test 2: Power Method WITH Sparsity Projection
fprintf('\n--- Test 2: With Sparsity-%d Projection ---\n', sparsity);
params2 = struct();
params2.is_matrix = false;
params2.sparsity = sparsity;
params2.projection = @sparsity_projection;

tic;
[Xl_init2, ~] = initialize_power_method(y, operator, d1, d2, T_power, params2);
t2 = toc;

error2 = norm(Xl_init2 - Xstar, 'fro') / norm(Xstar, 'fro');
sparsity2 = nnz(abs(Xl_init2) > 0.1 * max(abs(Xl_init2)));
recovered_support = find(abs(Xl_init2(:)) > 0.1 * max(abs(Xl_init2(:))));
support_match = length(intersect(support_idx, recovered_support));

fprintf('  Time: %.3f sec\n', t2);
fprintf('  Relative error: %.4e\n', error2);
fprintf('  Recovered sparsity: %d (true: %d)\n', sparsity2, sparsity);
fprintf('  Support recovery: %d/%d (%.1f%%)\n', support_match, sparsity, 100*support_match/sparsity);

%% Comparison
fprintf('\n--- Comparison ---\n');
fprintf('  Error reduction: %.2fx (%.4e -> %.4e)\n', error1/error2, error1, error2);
fprintf('  Projection benefit: %s\n', iif(error2 < error1, 'YES ✓', 'NO'));

%% Visualization
figure('Position', [100, 100, 1200, 800]);

subplot(3, 1, 1);
stem(Xstar, 'LineWidth', 1.5);
title('Ground Truth Sparse Signal');
ylabel('Amplitude');
grid on;
xlim([1, n]);

subplot(3, 1, 2);
stem(Xl_init1, 'LineWidth', 1.5);
title(sprintf('Without Projection (Error: %.2e, Sparsity: %d)', error1, sparsity1));
ylabel('Amplitude');
grid on;
xlim([1, n]);

subplot(3, 1, 3);
stem(Xl_init2, 'LineWidth', 1.5);
title(sprintf('With Sparsity-%d Projection (Error: %.2e, Support: %d/%d)', ...
    sparsity, error2, support_match, sparsity));
xlabel('Index');
ylabel('Amplitude');
grid on;
xlim([1, n]);

fprintf('\n=== Test Complete ===\n');

%% Helper Functions
function X_proj = sparsity_projection(X, params)
    % Hard thresholding: keep largest s elements
    s = params.sparsity;
    [~, idx] = sort(abs(X(:)), 'descend');
    X_proj = zeros(size(X));
    X_proj(idx(1:s)) = X(idx(1:s));
end

function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
