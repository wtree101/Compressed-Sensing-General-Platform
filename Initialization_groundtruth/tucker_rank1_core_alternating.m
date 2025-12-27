function [U_cells, q_opt, G_rank1, history] = tucker_rank1_core_alternating(operator, y, d1, d2, r, varargin)
% TUCKER_RANK1_CORE_ALTERNATING Alternating optimization for Tucker with rank-1 core
%
% 从 A^*(y) 提取 Tucker 张量 (r,r,r,r)，核心张量 G 在矩阵化为 (r²,r²) 后是秩-1的
%
% 方法：参数化 G_mat = q * q' (对称情况) 或 G_mat = p * q' (非对称)
%        交替优化 U_cells 和 q 来最小化 ||A(T) - y||^2
%
% Inputs:
%   operator - 测量算子结构体，包含 .A_cells
%   y        - 测量向量 (m × 1)
%   d1, d2   - 矩阵维度
%   r        - Tucker 秩
%   varargin - 可选参数：
%              'max_iter': 最大交替迭代次数 (默认: 20)
%              'tol': 收敛容差 (默认: 1e-6)
%              'verbose': 显示进度 (默认: true)
%              'symmetric': 对称 Tucker (U1=U2=U3=U4) (默认: false)
%              'init_method': 'hosvd' (HOSVD初始化) 或 'random' (默认: 'hosvd')
%              'u_iter': 每次 U 优化的内迭代次数 (默认: 5)
%              'q_iter': 每次 q 优化的内迭代次数 (默认: 10)
%              'mu_u': U 更新步长 (默认: 0.01)
%              'mu_q': q 更新步长 (默认: 0.1)
%
% Outputs:
%   U_cells  - 因子矩阵 {U1, U2, U3, U4}
%   q_opt    - 最优的 q 向量 (r² × 1)
%   G_rank1  - 秩-1 核心张量 (r×r×r×r)
%   history  - 收敛历史信息

    %% 解析参数
    p = inputParser;
    addParameter(p, 'max_iter', 20, @isnumeric);
    addParameter(p, 'tol', 1e-6, @isnumeric);
    addParameter(p, 'verbose', true, @islogical);
    addParameter(p, 'symmetric', false, @islogical);
    addParameter(p, 'init_method', 'hosvd', @ischar);
    addParameter(p, 'u_iter', 5, @isnumeric);
    addParameter(p, 'q_iter', 10, @isnumeric);
    addParameter(p, 'mu_u', 0.01, @isnumeric);
    addParameter(p, 'mu_q', 0.1, @isnumeric);
    parse(p, varargin{:});
    
    max_iter = p.Results.max_iter;
    tol = p.Results.tol;
    verbose = p.Results.verbose;
    use_symmetric = p.Results.symmetric;
    init_method = p.Results.init_method;
    u_iter = p.Results.u_iter;
    q_iter = p.Results.q_iter;
    mu_u = p.Results.mu_u;
    mu_q = p.Results.mu_q;
    
    m = length(y);
    
    if verbose
        fprintf('=== Tucker Tensor with Rank-1 Core (Alternating Optimization) ===\n');
        fprintf('Dimensions: d1=%d, d2=%d, Tucker rank=%d, Measurements: %d\n', d1, d2, r, m);
        fprintf('Max iterations: %d, Tolerance: %.2e\n', max_iter, tol);
        fprintf('Symmetric Tucker: %s\n', mat2str(use_symmetric));
        fprintf('U update: %d iters, step=%.3f | q update: %d iters, step=%.3f\n', ...
                u_iter, mu_u, q_iter, mu_q);
    end
    
    %% 步骤 1: 初始化
    if verbose
        fprintf('\n[Step 1] Initialization using %s method...\n', init_method);
    end
    
    % 创建 TuckerOperator
    if isa(operator, 'TuckerOperator')
        tucker_op = operator;
    else
        % 从 A_cells 创建
        A_cells = operator.A_cells;
        tucker_op = TuckerOperator(A_cells, 'order', 4, ...
                                   'symmetric', false, 'operator_type', 'kronecker');
    end
    
    switch lower(init_method)
        case 'hosvd'
            % 使用谱初始化
            y_spectral = y.^2 * sqrt(m);  % 转换为张量测量
            y_scaled = y_spectral / sqrt(m);
            H_tensor = tucker_op.kronecker_adjoint(y_scaled);
            
            % HOSVD 提取因子和核心
            rank_vec = [r, r, r, r];
            [G_init, U_cells_init] = HOSVD_with_factors(H_tensor, rank_vec);
            
            if use_symmetric
                U_cells = cell(1, 4);
                for i = 1:4
                    U_cells{i} = U_cells_init{1};
                end
            else
                U_cells = U_cells_init;
            end
            
            % 从 G_init 初始化 q
            G_mat_init = reshape(G_init, [r*r, r*r]);
            G_mat_sym = (G_mat_init + G_mat_init') / 2;
            [V, D] = eig(G_mat_sym);
            [~, idx] = max(abs(diag(D)));
            q = V(:, idx) * sqrt(abs(D(idx, idx)));
            
        case 'random'
            % 随机初始化
            U_cells = cell(1, 4);
            U_cells{1} = orth(randn(d1, r));
            U_cells{2} = orth(randn(d2, r));
            if use_symmetric
                U_cells{3} = U_cells{1};
                U_cells{4} = U_cells{2};
            else
                U_cells{3} = orth(randn(d1, r));
                U_cells{4} = orth(randn(d2, r));
            end
            q = randn(r*r, 1);
            q = q / norm(q);
            
        otherwise
            error('Unknown init_method: %s', init_method);
    end
    
    if verbose
        fprintf('  Initial ||q||_2 = %.6e\n', norm(q));
        fprintf('  Initial rank(G_mat) = %d (target: 1)\n', rank(q*q', 1e-10));
    end
    
    %% 步骤 2: 交替优化
    history = struct();
    history.loss = zeros(max_iter, 1);
    history.q_change = zeros(max_iter, 1);
    history.u_change = zeros(max_iter, 1);
    history.rank_G = zeros(max_iter, 1);
    
    if verbose
        fprintf('\n[Step 2] Alternating Optimization...\n');
        fprintf('Iter | Loss      | ||ΔU||   | ||Δq||   | rank(G)\n');
        fprintf('-----|-----------|----------|----------|--------\n');
    end
    
    for iter = 1:max_iter
        % 保存当前值用于收敛检查
        q_old = q;
        U_old = U_cells;
        
        %% 2a: 固定 q，优化 U (使用梯度下降)
        for u_step = 1:u_iter
            % 构造当前 Tucker 张量
            G_current = reshape(q * q', [r, r, r, r]);
            T_current = TuckerTensor([d1, d2, d1, d2], r, ...
                                     'symmetric', use_symmetric, ...
                                     'G', G_current, ...
                                     'U', U_cells);
            
            % 前向传播
            y_pred = tucker_op.forward(T_current) / sqrt(m);
            
            % 计算梯度（只更新 U，不更新 G）
            Grad_F = tucker_op.get_proj_grad_kronecker(T_current, y_pred/sqrt(m), y/sqrt(m));
            
            % 更新 U（简单梯度下降，保持正交性）
            for k = 1:4
                if use_symmetric && k > 1
                    break;  % 对称情况只更新 U{1}
                end
                
                % 梯度方向：Up{k}
                if ~isempty(Grad_F.Up{k})
                    U_new = U_cells{k} - mu_u * Grad_F.Up{k};
                    % 重正交化
                    U_cells{k} = orth(U_new);
                end
            end
            
            % 对称情况：复制 U{1}
            if use_symmetric
                for k = 2:4
                    U_cells{k} = U_cells{1};
                end
            end
        end
        
        %% 2b: 固定 U，优化 q (梯度下降在 q 空间)
        for q_step = 1:q_iter
            % 构造当前 Tucker 张量
            G_current = reshape(q * q', [r, r, r, r]);
            T_current = TuckerTensor([d1, d2, d1, d2], r, ...
                                     'symmetric', use_symmetric, ...
                                     'G', G_current, ...
                                     'U', U_cells);
            
            % 前向传播
            y_pred = tucker_op.forward(T_current);
            
            % 计算残差
            residual = y_pred / sqrt(m) - y;
            
            % 计算 q 的梯度
            % ∂Loss/∂q = ∂Loss/∂G : ∂G/∂q
            % 其中 G = reshape(q*q', [r,r,r,r])
            % ∂G/∂q 的计算需要链式法则
            
            % 获取关于 G 的梯度
            Grad_F = tucker_op.get_proj_grad_kronecker(T_current, y_pred/sqrt(m), y/sqrt(m));
            dG = Grad_F.G;  % r×r×r×r
            
            % 矩阵化 dG
            dG_mat = reshape(dG, [r*r, r*r]);
            
            % 链式法则: ∂Loss/∂q = 2 * dG_mat * q
            grad_q = 2 * dG_mat * q;
            
            % 梯度下降更新 q
            q = q - mu_q * grad_q;
            
            % 归一化 q（保持尺度一致性）
            q = q / norm(q);
        end
        
        %% 计算当前损失和收敛指标
        G_current = reshape(q * q', [r, r, r, r]);
        T_current = TuckerTensor([d1, d2, d1, d2], r, ...
                                 'symmetric', use_symmetric, ...
                                 'G', G_current, ...
                                 'U', U_cells);
        y_pred = tucker_op.forward(T_current) / sqrt(m);
        loss = 0.5 * norm(y_pred - y)^2;
        
        history.loss(iter) = loss;
        history.q_change(iter) = min(norm(q - q_old), norm(q + q_old));  % 考虑符号模糊性
        
        % U 的变化（只计算第一个，对称情况下代表所有）
        history.u_change(iter) = norm(U_cells{1} - U_old{1}, 'fro');
        
        % G_mat 的秩
        G_mat = q * q';
        history.rank_G(iter) = rank(G_mat, 1e-10);
        
        % 显示进度
        if verbose
            fprintf('%4d | %.4e | %.4e | %.4e | %d\n', ...
                    iter, loss, history.u_change(iter), ...
                    history.q_change(iter), history.rank_G(iter));
        end
        
        % 收敛检查
        if history.q_change(iter) < tol && history.u_change(iter) < tol
            if verbose
                fprintf('\n✓ Converged at iteration %d\n', iter);
            end
            % 截断历史记录
            history.loss = history.loss(1:iter);
            history.q_change = history.q_change(1:iter);
            history.u_change = history.u_change(1:iter);
            history.rank_G = history.rank_G(1:iter);
            break;
        end
    end
    
    %% 步骤 3: 输出结果
    q_opt = q;
    G_mat_final = q_opt * q_opt';
    G_rank1 = reshape(G_mat_final, [r, r, r, r]);
    
    if verbose
        fprintf('\n[Step 3] Final Results:\n');
        fprintf('  Final loss: %.6e\n', history.loss(end));
        fprintf('  Final ||q||_2: %.6e\n', norm(q_opt));
        fprintf('  rank(G_mat): %d (exact rank-1: %s)\n', ...
                rank(G_mat_final, 1e-10), ...
                mat2str(rank(G_mat_final, 1e-10) == 1));
        fprintf('  ||G_rank1||_F: %.6e\n', norm(G_rank1(:)));
        
        % 奇异值分析
        [~, S_G, ~] = svd(G_mat_final);
        s_vals = diag(S_G);
        fprintf('  Top 3 singular values of G_mat: [%.2e, %.2e, %.2e]\n', ...
                s_vals(1), s_vals(min(2,length(s_vals))), s_vals(min(3,length(s_vals))));
        if length(s_vals) > 1
            fprintf('  Singular value ratio σ_2/σ_1: %.2e (should be ~0 for rank-1)\n', ...
                    s_vals(2) / s_vals(1));
        end
    end
    
    if verbose
        fprintf('=== Alternating Optimization Complete ===\n\n');
    end
end

