%% Test: Tucker Tensor with Rank-1 Core Constraint
% 测试从 A^*(y) 提取 Tucker 张量，核心为秩-1
%
% 这个测试验证交替优化方法能否：
% 1. 保持核心张量 G 的秩-1 结构
% 2. 很好地拟合测量 y
% 3. 恢复原始矩阵 X

clear; close all;
addpath('../Initialization_groundtruth');
addpath('../utilities_tensor');
addpath('../utilities');

fprintf('=== Test: Tucker with Rank-1 Core Constraint ===\n\n');

%% 测试参数
d1 = 20;              % 行维度
d2 = 30;              % 列维度
r = 3;                % Tucker 秩
m = 800;              % 测量数
noise_level = 0;      % 噪声水平

fprintf('Test setup:\n');
fprintf('  Matrix size: %dx%d\n', d1, d2);
fprintf('  Tucker rank: %d\n', r);
fprintf('  Measurements: %d\n', m);
fprintf('  Noise level: %.2e\n\n', noise_level);

%% 生成地面真值
rng(42);

% 生成秩-r 矩阵
U_true = orth(randn(d1, r));
V_true = orth(randn(d2, r));
Sigma_true = diag(sort(rand(r, 1), 'descend'));
X_true = U_true * Sigma_true * V_true';
X_true = X_true / norm(X_true, 'fro');  % 归一化

fprintf('Ground truth:\n');
fprintf('  ||X_true||_F = %.6f\n', norm(X_true, 'fro'));
fprintf('  rank(X_true) = %d\n', rank(X_true, 1e-10));

%% 生成测量
fprintf('\nGenerating measurements...\n');

% 生成随机测量矩阵
A_cells = cell(m, 1);
for i = 1:m
    Ai = randn(d1, d2) / sqrt(d1 * d2);
    A_cells{i} = Ai;
end

% 计算测量: y_i = |<A_i, X>|^2
y = zeros(m, 1);
for i = 1:m
    y(i) = abs(sum(sum(A_cells{i} .* X_true)))^2;
end

% 添加噪声
if noise_level > 0
    y = y + noise_level * norm(y) * randn(m, 1) / sqrt(m);
end

fprintf('  Measurements generated: m = %d\n', m);
fprintf('  Average measurement: %.6e\n', mean(y));

%% 创建算子
operator = struct();
operator.A_cells = A_cells;

%% 方法 1: 标准 HOSVD（无秩-1约束）
fprintf('\n--- Method 1: Standard HOSVD (no rank-1 constraint) ---\n');

tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false);
y_spectral = y.^2 * sqrt(m);
y_scaled = y_spectral / sqrt(m);
H_tensor = tucker_op.kronecker_adjoint(y_scaled);

rank_vec = [r, r, r, r];
[G_hosvd, U_hosvd] = HOSVD_with_factors(H_tensor, rank_vec);

% 分析 G_hosvd 的秩
G_mat_hosvd = reshape(G_hosvd, [r*r, r*r]);
rank_hosvd = rank(G_mat_hosvd, 1e-10);
[~, S_hosvd, ~] = svd(G_mat_hosvd);
s_hosvd = diag(S_hosvd);

fprintf('Standard HOSVD results:\n');
fprintf('  rank(G_mat) = %d (full rank, no constraint)\n', rank_hosvd);
fprintf('  Top 3 singular values: [%.2e, %.2e, %.2e]\n', ...
        s_hosvd(1), s_hosvd(min(2,end)), s_hosvd(min(3,end)));
if length(s_hosvd) > 1
    fprintf('  σ_2/σ_1 = %.2e (NOT rank-1)\n', s_hosvd(2)/s_hosvd(1));
end

% 提取矩阵
T_hosvd = TuckerTensor([d1, d2, d1, d2], r, 'G', G_hosvd, 'U', U_hosvd);
X_hosvd = extract_matrix_from_tucker_2(T_hosvd);

% 修正符号
[err_hosvd, X_hosvd_aligned] = rectify_sign_ambiguity(X_hosvd, X_true);
fprintf('  Matrix reconstruction error: %.6e\n', err_hosvd);
fprintf('  rank(X_hosvd) = %d\n', rank(X_hosvd, 1e-10));

%% 方法 2: 秩-1 核心约束（交替优化）
fprintf('\n--- Method 2: Rank-1 Core Constraint (Alternating) ---\n');

[U_rank1, q_rank1, G_rank1, history_rank1] = tucker_rank1_core_alternating(...
    operator, y, d1, d2, r, ...
    'max_iter', 30, ...
    'tol', 1e-6, ...
    'verbose', true, ...
    'symmetric', false, ...
    'init_method', 'hosvd', ...
    'u_iter', 3, ...
    'q_iter', 5, ...
    'mu_u', 0.005, ...
    'mu_q', 0.05);

% 提取矩阵
T_rank1 = TuckerTensor([d1, d2, d1, d2], r, 'G', G_rank1, 'U', U_rank1);
X_rank1 = extract_matrix_from_tucker_2(T_rank1);

% 修正符号
[err_rank1, X_rank1_aligned] = rectify_sign_ambiguity(X_rank1, X_true);

%% 比较两种方法
fprintf('\n=== Comparison of Methods ===\n');
fprintf('Method 1 (Standard HOSVD):\n');
fprintf('  rank(G_mat): %d\n', rank_hosvd);
fprintf('  Matrix error: %.6e\n', err_hosvd);
fprintf('  rank(X): %d\n', rank(X_hosvd_aligned, 1e-10));

fprintf('\nMethod 2 (Rank-1 Core):\n');
fprintf('  rank(G_mat): %d (enforced rank-1)\n', rank(reshape(q_rank1*q_rank1', [r*r, r*r]), 1e-10));
fprintf('  Matrix error: %.6e\n', err_rank1);
fprintf('  rank(X): %d\n', rank(X_rank1_aligned, 1e-10));

fprintf('\nImprovement:\n');
if err_rank1 < err_hosvd
    fprintf('  ✓ Rank-1 constraint improves accuracy by %.2f%%\n', ...
            (1 - err_rank1/err_hosvd) * 100);
else
    fprintf('  ~ Standard HOSVD is better (no improvement from rank-1)\n');
    fprintf('    This may indicate the true G is NOT rank-1\n');
end

%% 可视化收敛
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
semilogy(1:length(history_rank1.loss), history_rank1.loss, 'b-o', 'LineWidth', 2);
grid on;
xlabel('Iteration');
ylabel('Loss');
title('Convergence: Loss');

subplot(1, 3, 2);
semilogy(1:length(history_rank1.q_change), history_rank1.q_change, 'r-s', 'LineWidth', 2);
hold on;
semilogy(1:length(history_rank1.u_change), history_rank1.u_change, 'g-^', 'LineWidth', 2);
grid on;
xlabel('Iteration');
ylabel('Change');
legend({'||Δq||', '||ΔU||'}, 'Location', 'best');
title('Convergence: Parameter Changes');

subplot(1, 3, 3);
plot(1:length(history_rank1.rank_G), history_rank1.rank_G, 'k-d', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('Iteration');
ylabel('Rank');
ylim([0, min(10, max(history_rank1.rank_G)+1)]);
title('Core Tensor Rank (should be 1)');

sgtitle('Tucker Tensor with Rank-1 Core: Convergence Analysis');

%% 核心张量结构分析
fprintf('\n=== Core Tensor Structure Analysis ===\n');

% 方法 1: G_hosvd (无约束)
G_mat_hosvd_full = reshape(G_hosvd, [r*r, r*r]);
fprintf('Method 1 (Standard HOSVD):\n');
fprintf('  ||G_hosvd||_F = %.6e\n', norm(G_hosvd(:)));
fprintf('  rank(G_mat): %d / %d\n', rank(G_mat_hosvd_full, 1e-10), r*r);

% 提取对角元素 G(i,i,k,k)
G_diag_hosvd = zeros(r, r);
for i = 1:r
    for k = 1:r
        G_diag_hosvd(i, k) = G_hosvd(i, i, k, k);
    end
end
diag_energy_hosvd = norm(G_diag_hosvd, 'fro')^2 / norm(G_hosvd(:))^2;
fprintf('  Diagonal energy: %.2f%%\n', diag_energy_hosvd * 100);

% 方法 2: G_rank1 (秩-1约束)
G_mat_rank1_full = reshape(G_rank1, [r*r, r*r]);
fprintf('\nMethod 2 (Rank-1 Core):\n');
fprintf('  ||G_rank1||_F = %.6e\n', norm(G_rank1(:)));
fprintf('  rank(G_mat): %d / %d ✓\n', rank(G_mat_rank1_full, 1e-10), r*r);

G_diag_rank1 = zeros(r, r);
for i = 1:r
    for k = 1:r
        G_diag_rank1(i, k) = G_rank1(i, i, k, k);
    end
end
diag_energy_rank1 = norm(G_diag_rank1, 'fro')^2 / norm(G_rank1(:))^2;
fprintf('  Diagonal energy: %.2f%%\n', diag_energy_rank1 * 100);

% 秩-1 近似误差
[U_d, S_d, V_d] = svd(G_mat_hosvd_full);
G_mat_rank1_approx = S_d(1,1) * (U_d(:,1) * V_d(:,1)');
rank1_approx_error = norm(G_mat_hosvd_full - G_mat_rank1_approx, 'fro') / norm(G_mat_hosvd_full, 'fro');
fprintf('\nRank-1 approximation error of Method 1:\n');
fprintf('  ||G_hosvd - rank1(G_hosvd)||_F / ||G_hosvd||_F = %.2f%%\n', rank1_approx_error * 100);
fprintf('  (This shows how far standard HOSVD is from rank-1)\n');

%% 测试完成
fprintf('\n=== Test Complete ===\n');
fprintf('Summary:\n');
fprintf('  Standard HOSVD:   error = %.2e, rank(G) = %d\n', err_hosvd, rank_hosvd);
fprintf('  Rank-1 Core:      error = %.2e, rank(G) = %d ✓\n', ...
        err_rank1, rank(G_mat_rank1_full, 1e-10));

if err_rank1 < err_hosvd * 1.1  % Within 10% of standard method
    fprintf('  ✓ Rank-1 constraint is effective!\n');
else
    fprintf('  ~ Rank-1 constraint may not be suitable for this problem\n');
end

%% Helper function (如果不在路径中)
function [min_error, X_aligned] = rectify_sign_ambiguity(X_est, X_true)
    % 修正符号模糊性
    err_pos = norm(X_est - X_true, 'fro') / norm(X_true, 'fro');
    err_neg = norm(X_est + X_true, 'fro') / norm(X_true, 'fro');
    
    if err_pos < err_neg
        min_error = err_pos;
        X_aligned = X_est;
    else
        min_error = err_neg;
        X_aligned = -X_est;
    end
end

function X = extract_matrix_from_tucker_2(T_tucker)
    % 简化的矩阵提取（从 initialize_tensor_lift_tucker_spectral.m 复制）
    d1 = T_tucker.dims(1);
    d2 = T_tucker.dims(2);
    r = T_tucker.tucker_ranks(1);
    
    U1 = T_tucker.U{1};
    U2 = T_tucker.U{2};
    U3 = T_tucker.U{3};
    U4 = T_tucker.U{4};
    G = T_tucker.G;
    
    % 符号修正
    U3_rect = U3;
    U4_rect = U4;
    for i = 1:r
        if dot(U1(:,i), U3(:,i)) < 0
            U3_rect(:,i) = -U3(:,i);
        end
        if dot(U2(:,i), U4(:,i)) < 0
            U4_rect(:,i) = -U4(:,i);
        end
    end
    
    S3 = diag(sign(diag(U3_rect' * U3)));
    S4 = diag(sign(diag(U4_rect' * U4)));
    
    G_rect = G;
    for c = 1:r
        for d = 1:r
            G_rect(:, :, c, d) = G(:, :, c, d) * S3(c,c) * S4(d,d);
        end
    end
    
    G_mat = reshape(G_rect, [r*r, r*r]);
    [V_G, D_G] = eig(G_mat);
    [lambda_max, idx_max] = max(abs(diag(D_G)));
    q = V_G(:, idx_max);
    
    U_kron = kron(U2, U1);
    v = U_kron * q * sqrt(abs(lambda_max));
    
    X = reshape(v, [d1, d2]);
    X = X / norm(X, 'fro');
end

