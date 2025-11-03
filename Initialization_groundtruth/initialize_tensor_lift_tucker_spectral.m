function [X0, U0, history] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params)
% INITIALIZE_TENSOR_LIFT_TUCKER_SPECTRAL Spectral Tucker tensor initialization
%
% This function performs SPECTRAL initialization using Tucker tensor decomposition
% with Riemannian Gradient Descent (RGD) on the Tucker manifold.
% Uses the formulation X = UU^T viewed as fourth-order Tucker tensor T.
%
% Key advantages:
%   - Spectral initialization: T = sum_i y_i * (Ai ⊗ Ai)
%   - Never forms full d×d×d×d tensor (memory efficient)
%   - Works in Tucker format: T = G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
%   - Uses Riemannian optimization on Tucker manifold
%   - Efficient operators via TuckerOperator class
%
% Inputs:
%   y        - Measurement vector (m x 1)
%   operator - Struct with fields:
%              .A: Forward operator @(X) A*X(:) for matrix
%              .A_star: Adjoint operator @(y) reshape(A'*y, [d1,d2])
%   d1       - Matrix row dimension
%   d2       - Matrix column dimension
%   params   - Struct with optional fields:
%              .T_power: Number of RGD iterations (default: 10)
%              .mu: Step size for RGD (default: 0.01)
%              .r: Tucker rank for tensor (default: min(5, d/4))
%              .Xstar: Ground truth for error tracking
%              .verbose: Print progress (default: false)
%                        Note: verbose=true automatically enables debug mode
%              .symmetric: Use symmetric Tucker (U₁=U₂=U₃=U₄) (default: false)
%              .pre_func: Preprocessing function for measurements (for spectral init)
%                         Example: @(y) set_zero_outside_range(y)
%              .precision_threshold: Switch to RGD when error <= threshold (optional)
%              .rgd_max_iter: Max iterations for RGD refinement (default: 100)
%              .rgd_mu: Step size for RGD refinement (default: same as mu)
%              .debug: Enable debug mode for tensor error tracking (default: false)
%
% Outputs:
%   X0       - Initialized matrix (d1 x d2) extracted from tensor
%   U0       - Factor matrix (empty, for compatibility)
%   history  - Struct with convergence information

    %% Validate symmetric case
    if d1 ~= d2
        error('Tensor lift initialization requires symmetric matrices: d1 must equal d2');
    end
    d = d1;
    m = length(y);
    
    %% Extract parameters
    T_power = get_param(params, 'T_power', 10);
    mu = get_param(params, 'mu', 0.01);
    
    % Tucker rank (default: adaptive based on problem size)
    if isfield(params, 'r')
        tucker_rank = params.r;
    else
        tucker_rank = min(5, max(1, floor(d/4)));
    end
    
    use_symmetric = get_param(params, 'symmetric', false);
    verbose = get_param(params, 'verbose', false);
    debug_mode = get_param(params, 'debug', false);
    
    % If verbose is on, automatically enable debug mode
    if verbose
        debug_mode = true;
    end
    
    has_ground_truth = isfield(params, 'Xstar') && ~isempty(params.Xstar);
    
    % Precision threshold for switching to RGD (optional)
    precision_threshold = get_param(params, 'precision_threshold', []);
    use_rgd_refinement = ~isempty(precision_threshold);
    
    % RGD parameters (if using RGD refinement)
    rgd_max_iter = get_param(params, 'rgd_max_iter', 100);
    rgd_mu = get_param(params, 'rgd_mu', mu);
    
    if verbose
        fprintf('=== Tucker-based Tensor Lift Initialization (Spectral) ===\n');
        fprintf('Matrix: %dx%d, Tucker rank: %d, Measurements: %d\n', d, d, tucker_rank, m);
        fprintf('RGD iterations: %d, Step size: %.4f\n', T_power, mu);
        fprintf('Symmetric Tucker: %s\n', mat2str(use_symmetric));
        fprintf('Debug mode: ON (auto-enabled by verbose mode)\n');
        if use_rgd_refinement
            fprintf('RGD refinement: ON (threshold=%.2e, max_iter=%d)\n', precision_threshold, rgd_max_iter);
        end
    end
    
    %% Initialize history
    history = struct();
    history.method = 'tensor_lift_tucker';
    history.iterations = T_power;
    history.loss_function = zeros(T_power, 1);
    
    if has_ground_truth
        Xstar = params.Xstar;
        history.matrix_errors = zeros(T_power, 1);
    end
    
    % Debug mode: create ground truth tensor and compute tensor losses
    if debug_mode && has_ground_truth
        if verbose
            fprintf('\n[DEBUG] Creating ground truth tensor Tstar = Xstar ⊗ Xstar...\n');
        end
        
        % Form ground truth tensor: Tstar = Xstar ⊗ Xstar
        % This is a d×d×d×d tensor
        Tstar_full = zeros(d, d, d, d);
        for i = 1:d
            for j = 1:d
                for k = 1:d
                    for l = 1:d
                        Tstar_full(i,j,k,l) = Xstar(i,j) * Xstar(k,l);
                    end
                end
            end
        end
        
        % Compute norm for normalization
        Tstar_norm = norm(Tstar_full(:));
        
        if verbose
            fprintf('[DEBUG] Ground truth tensor created: size [%s], norm=%.6f\n', ...
                    num2str(size(Tstar_full)), Tstar_norm);
        end
        
        % Store for loss computation
        history.Tstar_full = Tstar_full;
        history.Tstar_norm = Tstar_norm;
        history.tensor_errors = zeros(T_power, 1);
        history.tensor_errors_relative = zeros(T_power, 1);
    end
    
    %% Extract measurement matrices from operator
    if verbose
        fprintf('Extracting measurement matrices...\n');
    end
    
    n = d * d;
    A_matrix = zeros(m, n);
    for j = 1:n
        e_j = zeros(n, 1);
        e_j(j) = 1;
        E_j = reshape(e_j, [d, d]);
        A_matrix(:, j) = operator.A(E_j);
    end
    
    % Create cell array of measurement matrices
    A_cells = cell(m, 1);
    for i = 1:m
        Ai = reshape(A_matrix(i, :), [d, d]);
        A_cells{i} = (Ai + Ai') / 2;  % Symmetrize
    end
    
    %% Create Tucker operator (never forms A_i ⊗ A_i explicitly)
    tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false);
    tucker_op.A_mat = A_matrix';  % Store for efficient computation (d² × m)
    
    if verbose
        fprintf('Tucker operator created (Kronecker structure)\n');
    end
    
    %% Initialize Tucker tensor using spectral method
    dims = [d, d, d, d];
    
    % Spectral initialization: directly form T = sum_i y_i * (Ai ⊗ Ai)
    if verbose
        fprintf('Using spectral initialization (direct tensor formation)...\n');
    end
    
    % Create TuckerTensor object
    T_tucker = TuckerTensor(dims, tucker_rank, ...
                            'symmetric', use_symmetric, ...
                            'init_method', 'zeros', ...
                            'debug', debug_mode);
    
    % Prepare operator struct for spectral initialization
    spectral_operator = struct();
    spectral_operator.A_cells = A_cells;
    
    % Convert phase retrieval measurements to tensor measurements
    % For spectral init: y_spectral_i = (<Ai, X>)^2 / sqrt(m) = y_i^2 * sqrt(m)
    y_spectral = y.^2 * sqrt(m);
    
    % Get preprocessing function if provided
    if isfield(params, 'pre_func') && ~isempty(params.pre_func)
        pre_func = params.pre_func;
        if verbose
            fprintf('Spectral init: using provided preprocessing function\n');
        end
    else
        pre_func = @(y) y;  % Identity (no preprocessing)
    end
    
    % Call initialize_spectral with preprocessing
    [U_cell_init, G_init] = T_tucker.initialize_spectral(spectral_operator, y_spectral, m, 'pre_func', pre_func);
    
    % Set the initialized factor matrices and core
    for k = 1:4
        T_tucker.U{k} = U_cell_init{k};
    end
    T_tucker.G = G_init;
    
    if verbose
        fprintf('Spectral initialization complete: core norm = %.6f\n', norm(T_tucker.G(:)));
        fprintf('  Factor matrix U: (%d x %d), condition: %.2e\n', ...
                size(U_cell_init{1}, 1), size(U_cell_init{1}, 2), cond(U_cell_init{1}));
    end
    
    %% Riemannian Gradient Descent on Tucker manifold
    if verbose
        fprintf('\nRunning Riemannian Gradient Descent...\n');
        if debug_mode
            fprintf('Iter | Loss      | Tensor Err | Matrix Err | Step\n');
            fprintf('-----|-----------|------------|------------|-----\n');
        else
            fprintf('Iter | Loss      | Matrix Err | Step\n');
            fprintf('-----|-----------|------------|-----\n');
        end
    end
    
    for t = 1:T_power
        % Forward pass: compute measurements
        y_pred = tucker_op.forward(T_tucker) / sqrt(m);
        
        % Compute loss
        residual = y_pred - y;
        loss = 0.5 * norm(residual)^2;
        history.loss_function(t) = loss;
        
        % Compute Riemannian gradient
        Grad_F = tucker_op.get_proj_grad_kronecker(T_tucker, y_pred/ sqrt(m), y/ sqrt(m)) ;
        
        % Debug mode: Compare retraction vs HOSVD
        if debug_mode   % Only for first few iterations (expensive)
            if verbose && t == 1
                fprintf('\n[DEBUG] Comparing retraction vs full HOSVD approach...\n\n');
            end
            
            % Save tensors for external testing (first iteration only)
            if t == 1
                % Save T_tucker and Grad_F to file
                debug_data = struct();
                debug_data.T_tucker = T_tucker;
                debug_data.Grad_F = Grad_F;
                debug_data.eta = mu;
                debug_data.tucker_rank = tucker_rank;
                debug_data.dims = dims;
                save('debug_tucker_tensors.mat', 'debug_data');
                if verbose
                    fprintf('[DEBUG] Saved T_tucker and Grad_F to debug_tucker_tensors.mat\n');
                    fprintf('[DEBUG] You can now test with: test_tucker_retraction_from_file.m\n\n');
                end
            end
            
            % Method 1: Retraction (existing method)
            T_retract = T_tucker.retraction(Grad_F, mu);
            
            % Method 2: Full gradient + HOSVD
            % Form T_full - mu * Grad_F_full
            T_current_full = T_tucker.full();

            % Debug
            % First term: Grad_F.G ×₁ U₁ ×₂ U₂ ×₃ ... ×_N U_N
            if isscalar(Grad_F.G)
                % Special case: rank-1 tensor (scalar core)
                % Form as Kronecker product: U₁ ⊗ U₂ ⊗ ... ⊗ U_N
                Grad_full = T_tucker.U{1};
                for i = 2:length(dims)
                    Grad_full = kron(Grad_full, T_tucker.U{i});
                end
                Grad_full = reshape(Grad_full, dims) * Grad_F.G;
            else
                % General case: apply mode products
                Grad_full = Grad_F.G;
                for i = 1:length(dims)
                    Grad_full = tensor_mode_product(Grad_full, T_tucker.U{i}, i);
                end
            end

            % Add terms for each mode: G ×₁ U₁ ... ×ᵢ Upᵢ ... ×_N U_N
            for i = 1:length(dims)
                if isscalar(T_tucker.G)
                    % Special case: rank-1 tensor (scalar core)
                    term = T_tucker.U{1};
                    for j = 2:length(dims)
                        if j == i
                            term = kron(term, Grad_F.Up{j});
                        else
                            term = kron(term, T_tucker.U{j});
                        end
                    end
                    % Handle first mode
                    if i == 1
                        term_temp = Grad_F.Up{1};
                        for j = 2:length(dims)
                            term_temp = kron(term_temp, T_tucker.U{j});
                        end
                        term = reshape(term_temp, dims) * T_tucker.G;
                    else
                        term = reshape(term, dims) * T_tucker.G;
                    end
                else
                    % General case: apply mode products
                    term = T_tucker.G;
                    for j = 1:length(dims)
                        if j == i
                            term = tensor_mode_product(term, Grad_F.Up{j}, j);
                        else
                            term = tensor_mode_product(term, T_tucker.U{j}, j);
                        end
                    end
                end
                Grad_full = Grad_full + term;
            end
            T_updated_full = T_current_full - mu * Grad_full;
            
            % HOSVD truncation to Tucker rank
            T_hosvd = HOSVD(T_updated_full, tucker_rank*[1,1,1,1]);
            
            % Compare results
            T_retract_full = T_retract.full();
            diff_methods = T_retract_full - T_hosvd;
            rel_diff = norm(diff_methods(:)) / norm(T_retract_full(:));
            
            if verbose
                fprintf('[DEBUG] Iter %d: Retraction vs HOSVD difference = %.6e\n', t, rel_diff);
            end
        else
            % Normal retraction without comparison
            T_tucker = T_tucker.retraction(Grad_F, mu);
        end
        
        % Update T_tucker for next iteration
        if debug_mode 
            T_tucker = T_retract;  % Use retraction result
        end
        
        % Debug mode: compute tensor error
        if debug_mode && has_ground_truth
            % Reconstruct full tensor from Tucker format
            T_current = T_tucker.full();
            
            % Compute tensor errors
            diff_tensor = T_current - history.Tstar_full;
            tensor_err_abs = norm(diff_tensor(:));
            tensor_err_rel = tensor_err_abs / history.Tstar_norm;
            
            history.tensor_errors(t) = tensor_err_abs;
            history.tensor_errors_relative(t) = tensor_err_rel;
        end
        
        % Extract matrix and compute error if ground truth available
        if has_ground_truth
            X_current = extract_matrix_from_tucker(T_tucker);
            X_current = (X_current + X_current') / 2;  % Symmetrize
            [history.matrix_errors(t), ~] = rectify_sign_ambiguity(X_current, Xstar);
            
            
            
            if verbose && (mod(t, max(1, floor(T_power/10))) == 0 || t == 1)
                if debug_mode
                    fprintf('%4d | %.4e | %.4e | %.4e | %.3f\n', ...
                        t, loss, history.tensor_errors_relative(t), ...
                        history.matrix_errors(t), mu);
                else
                    fprintf('%4d | %.4e | %.4e | %.3f\n', ...
                        t, loss, history.matrix_errors(t), mu);
                end
            end
        else
            % Check loss-based threshold if no ground truth
            if use_rgd_refinement && loss <= precision_threshold
                if verbose
                    fprintf('\n[Loss threshold reached: %.2e <= %.2e]\n', ...
                            loss, precision_threshold);
                    fprintf('Switching to RGD refinement...\n');
                end
                
                % Run RGD refinement
                rgd_params = struct();
                rgd_params.T = rgd_max_iter;
                rgd_params.mu = rgd_mu;
                rgd_params.r = tucker_rank;
                rgd_params.verbose = verbose;
                
                [rgd_output, T_tucker] = solve_RGD_tucker(T_tucker, y, tucker_op, rgd_params);
                
                % Update history
                history.rgd_used = true;
                history.rgd_iterations = rgd_max_iter;
                history.rgd_loss = rgd_output.Error_function;
                
                % Extract final matrix after RGD
                X0 = extract_matrix_from_tucker(T_tucker);
                X0 = (X0 + X0') / 2;
                
                if verbose
                    fprintf('RGD refinement complete: Final loss = %.6e\n', rgd_output.Error_function(end));
                end
                
                % Skip remaining iterations
                break;
            end
            
            if verbose && (mod(t, max(1, floor(T_power/10))) == 0 || t == 1)
                fprintf('%4d | %.4e | --         | %.3f\n', t, loss, mu);
            end
        end
    end
    
    %% Extract final matrix from Tucker tensor
    if verbose
        fprintf('\nExtracting final matrix from Tucker tensor...\n');
    end
    
    % Only extract if RGD wasn't used (already extracted above)
    if ~isfield(history, 'rgd_used') || ~history.rgd_used
        X0 = extract_matrix_from_tucker(T_tucker);
        X0 = (X0 + X0') / 2;  % Symmetrize
    end
    
    %% Final error and output
    U0 = [];  % For compatibility
    
    if has_ground_truth
        [final_error, X0] = rectify_sign_ambiguity(X0, Xstar);
        history.final_error = final_error;
        
        if verbose
            fprintf('Final matrix error: %.6e\n', final_error);
            fprintf('Final matrix rank: %d (Tucker rank: %d)\n', rank(X0, 1e-6), tucker_rank);
            
            if debug_mode
                fprintf('[DEBUG] Final tensor error (relative): %.6e\n', ...
                        history.tensor_errors_relative(end));
                fprintf('[DEBUG] Final tensor error (absolute): %.6e\n', ...
                        history.tensor_errors(end));
            end
        end
    end
    
    if verbose
        fprintf('=== Tucker Tensor Lift Initialization Complete ===\n\n');
    end
end

%% Helper Functions

function value = get_param(params, field, default)
    % Get parameter from struct with default value
    if isfield(params, field)
        value = params.(field);
    else
        value = default;
    end
end

function X = extract_matrix_from_tucker(T_tucker)
    % EXTRACT_MATRIX_FROM_TUCKER Extract matrix from Tucker tensor
    % For 4th-order tensor T = X ⊗ X, extract X via matricization
    %
    % Method: Matricize T to (d² × d²) and extract leading eigenvector
    
    d = T_tucker.dims(1);
    r = T_tucker.tucker_ranks(1);
    
    % Compute T in matricized form (d² × d²) without forming full tensor
    % T_mat = (U₁ ⊗ U₂) * G_mat * (U₃ ⊗ U₄)'
    % where G_mat is r² × r² matricization of core
    
    U1 = T_tucker.U{1};
    U2 = T_tucker.U{2};
    U3 = T_tucker.U{3};
    U4 = T_tucker.U{4};
    G = T_tucker.G;
    
    % Handle rank-1 case (scalar core)
    if isscalar(G)
        % For scalar core, form Kronecker products directly
        U_left = kron(U1, U2);   % d² × r
        U_right = kron(U3, U4);  % d² × r
        T_mat = G * (U_left * U_right');
    else
        % General case: form matricized core and compute
        G_mat = reshape(permute(G, [1,2,3,4]), [r*r, r*r]);
        
        % Form Kronecker products of factors
        U_left = kron(U1, U2);   % d² × r²
        U_right = kron(U3, U4);  % d² × r²
        
        % Matricized tensor: T_mat = U_left * G_mat * U_right'
        T_mat = U_left * G_mat * U_right';
    end
    
    % Symmetrize
    T_mat = (T_mat + T_mat') / 2;
    
    % Extract leading eigenvector and reshape to matrix
    [V, D] = eig(T_mat);
    [~, idx] = max(abs(diag(D)));
    v_lead = V(:, idx);
    
    % Reshape to matrix
    X = reshape(v_lead, [d, d]);
    
    % Normalize
    X = X / norm(X, 'fro');
    
    % Make symmetric
    X = (X + X') / 2;
end



function T_mat = mode_n_unfold(T, n, dims)
    % MODE_N_UNFOLD Mode-n unfolding (matricization) of tensor
    % Returns matrix of size (dims(n) × prod(dims([1:n-1, n+1:end])))
    
    N = length(dims);
    perm = [n, 1:n-1, n+1:N];
    T_perm = permute(T, perm);
    T_mat = reshape(T_perm, dims(n), []);
end

function T_out = tensor_mode_product_helper(T, M, mode)
    % TENSOR_MODE_PRODUCT_HELPER n-mode product helper function
    % T_out = T ×_mode M
    
    sz = size(T);
    k = size(M, 1);
    
    % Permute so mode is first
    order = 1:max(ndims(T), mode);
    order([1, mode]) = [mode, 1];
    T_perm = permute(T, order);
    
    % Reshape to matrix and multiply
    T_mat = reshape(T_perm, sz(mode), []);
    T_out_mat = M * T_mat;
    
    % Reshape back
    sz_out = sz;
    sz_out(mode) = k;
    sz_out_perm = sz_out(order);
    T_out_perm = reshape(T_out_mat, sz_out_perm);
    
    % Permute back
    T_out = ipermute(T_out_perm, order);
end
