%% Quick Test: Core Projection Impact
% Quick comparison of WITH vs WITHOUT core projection
% Run this for a fast check before the comprehensive test

clear; clc;
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('Quick Test: Core Projection Impact\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

%% Setup
d1 = 20; d2 = 30;  % Non-square matrix dimensions  
r = 3; r_tucker = r; m = 1100;
mu = 0.1; T = 200;
use_abs = true;
rng(45);

fprintf('Config: d1=%d, d2=%d, r=%d, m=%d, T=%d\n', d1, d2, r, m, T);
if use_abs
    fprintf('Ground truth: with abs()\n\n');
else
    fprintf('Ground truth: without abs()\n\n');
end

%% Generate Problem
U1_true = randn(d1, r);
U2_true = randn(d2, r);
if use_abs
    Xstar = abs(U1_true) * abs(U2_true)';
else
    Xstar = U1_true * U2_true';
end
Xstar = Xstar / norm(Xstar, 'fro');

n = d1 * d2;
A = randn(m, n);
operator = struct();
operator.A = @(X) A * X(:);
operator.A_star = @(y) reshape(A' * y, [d1, d2]);

y = abs(operator.A(Xstar)) / sqrt(m);

fprintf('✓ Problem generated\n');

%% Setup Tucker Operator
A_matrix = zeros(m, n);
for j = 1:n
    e_j = zeros(n, 1);
    e_j(j) = 1;
    E_j = reshape(e_j, [d1, d2]);
    A_matrix(:, j) = operator.A(E_j);
end

A_cells = cell(m, 1);
for i = 1:m
    Ai = reshape(A_matrix(i, :), [d1, d2]);
    A_cells{i} = Ai;
end

tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false);
tucker_op.A_mat = A_matrix';

fprintf('✓ Tucker operator created\n');

%% Spectral Initialization
dims = [d1, d2, d1, d2];
T_tucker_init = TuckerTensor(dims, r, 'symmetric', false, 'init_method', 'zeros');

spectral_operator = struct();
spectral_operator.A_cells = A_cells;
y_spectral = y.^2 * sqrt(m);

[U_cell_init, G_init] = T_tucker_init.initialize_spectral(spectral_operator, y_spectral, m);

for k = 1:4
    T_tucker_init.U{k} = U_cell_init{k};
end
T_tucker_init.G = G_init;

fprintf('✓ Spectral initialization complete\n\n');

%% Test WITHOUT Projection
fprintf('Running WITHOUT core projection...\n');
params_no = struct('T', T, 'mu', mu, 'Xstar', Xstar, 'verbose', false, 'use_core_projection', false);
T_tucker_no = T_tucker_init.copy();

tic;
[output_no, ~] = solve_RGD_tucker_kronecker(T_tucker_no, [], y_spectral, tucker_op, [], [], [], m, params_no);
time_no = toc;

error_no = output_no.Error_Stand(end);
loss_no = output_no.Error_function(end);

fprintf('  Final Error: %.6e\n', error_no);
fprintf('  Final Loss:  %.6e\n', loss_no);
fprintf('  Time:        %.2f seconds\n\n', time_no);

%% Test WITH Projection (using solver option)
fprintf('Running WITH core projection (use_core_projection=true)...\n');
params_with = struct('T', T, 'mu', mu, 'Xstar', Xstar, 'verbose', false, 'use_core_projection', true);
T_tucker_with = T_tucker_init.copy();

tic;
[output_with, ~] = solve_RGD_tucker_kronecker(T_tucker_with, [], y_spectral, tucker_op, [], [], [], m, params_with);
time_with = toc;

error_with = output_with.Error_Stand(end);
loss_with = output_with.Error_function(end);

fprintf('  Final Error: %.6e\n', error_with);
fprintf('  Final Loss:  %.6e\n', loss_with);
fprintf('  Time:        %.2f seconds\n\n', time_with);

%% Comparison
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('Comparison Results\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

fprintf('%-30s | %-15s | %-15s\n', 'Metric', 'Without Proj', 'With Proj');
fprintf('%-30s-|%-15s-|%-15s\n', repmat('-', 1, 30), repmat('-', 1, 15), repmat('-', 1, 15));
fprintf('%-30s | %.6e    | %.6e\n', 'Final Error', error_no, error_with);
fprintf('%-30s | %.6e    | %.6e\n', 'Final Loss', loss_no, loss_with);
fprintf('%-30s | %.2f s         | %.2f s\n', 'Computation Time', time_no, time_with);

fprintf('\n');

% Calculate improvement
if error_with < error_no
    improvement = (error_no - error_with) / error_no * 100;
    fprintf('✓✓✓ Core projection IMPROVES error by %.2f%%\n', improvement);
    fprintf('    Recommendation: ENABLE projection (uncomment line 135)\n');
elseif error_with > error_no
    degradation = (error_with - error_no) / error_no * 100;
    fprintf('✗✗✗ Core projection DEGRADES error by %.2f%%\n', degradation);
    fprintf('    Recommendation: Keep projection DISABLED\n');
else
    fprintf('→ No significant difference\n');
end

fprintf('\n');

% Convergence speed
threshold = 1e-3;
iter_no = find(output_no.Error_Stand < threshold, 1);
iter_with = find(output_with.Error_Stand < threshold, 1);

if ~isempty(iter_no) && ~isempty(iter_with)
    fprintf('Convergence to %.0e error:\n', threshold);
    fprintf('  Without projection: %d iterations\n', iter_no);
    fprintf('  With projection:    %d iterations\n', iter_with);
    if iter_with < iter_no
        speedup = (iter_no - iter_with) / iter_no * 100;
        fprintf('  → Projection accelerates convergence by %.1f%%\n', speedup);
    end
end

fprintf('\n');

%% Visualization
fprintf('Generating comparison plot...\n');

figure('Position', [100, 100, 1400, 500]);

subplot(1, 3, 1);
semilogy(output_no.Error_Stand, 'b-', 'LineWidth', 2, 'DisplayName', 'Without Projection');
hold on;
semilogy(output_with.Error_Stand, 'r--', 'LineWidth', 2, 'DisplayName', 'With Projection');
xlabel('Iteration');
ylabel('Relative Error');
title('Error Convergence');
legend('Location', 'best');
grid on;

subplot(1, 3, 2);
semilogy(output_no.Error_function, 'b-', 'LineWidth', 2, 'DisplayName', 'Without Projection');
hold on;
semilogy(output_with.Error_function, 'r--', 'LineWidth', 2, 'DisplayName', 'With Projection');
xlabel('Iteration');
ylabel('Loss');
title('Loss Convergence');
legend('Location', 'best');
grid on;

subplot(1, 3, 3);
error_diff = abs(output_no.Error_Stand - output_with.Error_Stand);
semilogy(error_diff, 'k-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Absolute Difference');
title('|Error_{no\_proj} - Error_{with\_proj}|');
grid on;

sgtitle(sprintf('Core Projection Impact (m=%d, d1×d2=%d×%d, r=%d)', m, d1, d2, r));

fprintf('✓ Plot generated\n\n');
fprintf('✓ Quick Test Complete\n');


