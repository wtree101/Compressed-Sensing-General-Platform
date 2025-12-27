function [X0, U0, history] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params)
% INITIALIZE_TENSOR_LIFT_TUCKER_SPECTRAL Spectral Tucker tensor initialization
%
% This function performs SPECTRAL initialization using Tucker tensor decomposition
% with Riemannian Gradient Descent (RGD) on the Tucker manifold.
% Uses the formulation X = UU^T viewed as fourth-order Tucker tensor T.
%
% Key advantages:
%   - Spectral initialization: T = sum_i y_i * (Ai ⊗ Ai)
%   - Never forms full d1×d2×d1×d2 tensor (memory efficient)
%   - Works in Tucker format: T = G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
%   - Uses Riemannian optimization on Tucker manifold
%   - Efficient operators via TuckerOperator class
%   - Supports both square (d1=d2) and non-square (d1≠d2) matrices
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
%              .r: Tucker rank for tensor (default: min(5, min(d1,d2)/4))
%              .Xstar: Ground truth for error tracking
%              .verbose: Print progress (default: false)
%                        Note: verbose=true automatically enables debug mode
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

    %% Setup dimensions
    m = length(y);
    is_square = (d1 == d2);
    
    %% Extract parameters
    T_power = get_param(params, 'T_power', 10);
    mu = get_param(params, 'mu', 0.01);
    
    % Tucker rank (default: adaptive based on problem size)
    if isfield(params, 'r')
        tucker_rank = params.r;
    else
        tucker_rank = min(5, max(1, floor(min(d1, d2)/4)));
    end
    
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
        fprintf('Matrix: %dx%d, Tucker rank: %d, Measurements: %d\n', d1, d2, tucker_rank, m);
        fprintf('RGD iterations: %d, Step size: %.4f\n', T_power, mu);
        fprintf('Square matrix: %s\n', mat2str(is_square));
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
        % For non-symmetric: d1×d2×d1×d2 tensor
        Tstar_full = zeros(d1, d2, d1, d2);
        for i = 1:d1
            for j = 1:d2
                for k = 1:d1
                    for l = 1:d2
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
    
    n = d1 * d2;
    A_matrix = zeros(m, n);
    for j = 1:n
        e_j = zeros(n, 1);
        e_j(j) = 1;
        E_j = reshape(e_j, [d1, d2]);
        A_matrix(:, j) = operator.A(E_j);
    end
    
    % Create cell array of measurement matrices
    A_cells = cell(m, 1);
    for i = 1:m
        Ai = reshape(A_matrix(i, :), [d1, d2]);
        A_cells{i} = Ai;  % No symmetrization
    end
    
    %% Create Tucker operator (never forms A_i ⊗ A_i explicitly)
    tucker_op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false);
    tucker_op.A_mat = A_matrix';  % Store for efficient computation (n × m)
    
    if verbose
        fprintf('Tucker operator created (Kronecker structure)\n');
    end
    
    %% Initialize Tucker tensor using spectral method
    % For matrices: d1×d2×d1×d2 tensor (no symmetry assumption)
    dims = [d1, d2, d1, d2];
    
    % Spectral initialization: directly form T = sum_i y_i * (Ai ⊗ Ai)
    if verbose
        fprintf('Using spectral initialization (direct tensor formation)...\n');
    end
    
    % Create TuckerTensor object (no symmetry)
    T_tucker = TuckerTensor(dims, tucker_rank, ...
                            'symmetric', false, ...
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
    
    %% Riemannian Gradient Descent on Tucker manifold using solver
    if verbose
        fprintf('\nRunning Riemannian Gradient Descent (via solve_RGD_tucker_kronecker)...\n');
    end
    
    % Prepare parameters for solver
    solver_params = struct();
    solver_params.T = T_power;
    solver_params.mu = mu;
    solver_params.verbose = verbose;
    solver_params.use_core_projection = false;
    if has_ground_truth
        solver_params.Xstar = Xstar;
    end
    
    % Call the solver (use y_spectral, not y!)
    [solver_output, T_tucker] = solve_RGD_tucker_kronecker(T_tucker, [], y_spectral, tucker_op, [], [], [], m, solver_params);
    
    % Update history with solver output
    history.loss_function = solver_output.Error_function;
    if has_ground_truth
        history.matrix_errors = solver_output.Error_Stand;
    end
    
    %% Extract final matrix from Tucker tensor
    if verbose
        fprintf('\nExtracting final matrix from Tucker tensor...\n');
    end
    
    X0 = extract_matrix_from_tucker_2(T_tucker);  % Use NEW efficient method (no symmetrization)
    
    %% Final error and output
    U0 = [];  % For compatibility
    
    if has_ground_truth
        [final_error, X0] = rectify_sign_ambiguity(X0, Xstar);
        history.final_error = final_error;
        
        if verbose
            fprintf('Final matrix error: %.6e\n', final_error);
            fprintf('Final matrix rank: %d (Tucker rank: %d)\n', rank(X0, 1e-6), tucker_rank);
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

function X = extract_matrix_from_tucker_2(T_tucker)
    % EXTRACT_MATRIX_FROM_TUCKER_2 Extract matrix from Tucker tensor (NEW DEFAULT METHOD)
    % For 4th-order tensor T = X ⊗ X, extract X via core tensor eigendecomposition
    %
    % Method 2 (NEW, EFFICIENT): 
    %   1. Rectify sign ambiguity: U1≈U3, U2≈U4
    %   2. Form G_mat (r²×r²) from core tensor G (r×r×r×r)
    %   3. Eigendecompose G_mat to get leading eigenvector q
    %   4. Reconstruct v = (U2⊗U1) * q
    %   5. Reshape v to get X (d1×d2)
    %
    % Advantages over Method 1:
    %   - Eigendecompose r²×r² instead of (d1*d2)×(d1*d2)
    %   - Much faster when r << d1*d2
    %   - Numerically more stable
    %
    % For non-square matrices: d1×d2×d1×d2 tensor
    
    d1 = T_tucker.dims(1);
    d2 = T_tucker.dims(2);
    r = T_tucker.tucker_ranks(1);
    
    % Extract factor matrices and core
    U1 = T_tucker.U{1};  % d1 × r
    U2 = T_tucker.U{2};  % d2 × r
    U3 = T_tucker.U{3};  % d1 × r
    U4 = T_tucker.U{4};  % d2 × r
    G = T_tucker.G;      % r × r × r × r
    
    % Step 1: Rectify sign ambiguity to enforce U1≈U3, U2≈U4
    % For each column, align signs
    U3_rect = U3;
    U4_rect = U4;
    
    for i = 1:r
        % Align U3(:,i) with U1(:,i)
        if dot(U1(:,i), U3(:,i)) < 0
            U3_rect(:,i) = -U3(:,i);
        end
        % Align U4(:,i) with U2(:,i)
        if dot(U2(:,i), U4(:,i)) < 0
            U4_rect(:,i) = -U4(:,i);
        end
    end
    
    % Recompute core tensor with rectified factors
    % G_rect = H ×₁ U1' ×₂ U2' ×₃ U3_rect' ×₄ U4_rect'
    % We need to adjust G accordingly
    % Since we only flipped signs, we can construct sign correction matrices
    S3 = diag(sign(diag(U3_rect' * U3)));  % Sign correction for mode 3
    S4 = diag(sign(diag(U4_rect' * U4)));  % Sign correction for mode 4
    
    % Apply sign corrections to core: G_rect(a,b,c,d) = G(a,b,c,d) * S3(c,c) * S4(d,d)
    G_rect = G;
    for c = 1:r
        for d = 1:r
            G_rect(:, :, c, d) = G(:, :, c, d) * S3(c,c) * S4(d,d);
        end
    end
    
    % Step 2: Form G_mat (r² × r²) from G_rect
    % Matricize: modes (1,2) vs modes (3,4)
    G_mat = reshape(G_rect, [r*r, r*r]);
    
    % Note: G_mat may not be symmetric for non-square matrices (d1≠d2)
    % This is expected and handled by eigendecomposition
    
    % Step 3: Eigendecompose G_mat (r² × r²) - much smaller than full tensor!
    [V_G, D_G] = eig(G_mat);
    [lambda_max, idx_max] = max(abs(diag(D_G)));
    q = V_G(:, idx_max);  % Leading eigenvector (r² × 1)
    
    % Step 4: Reconstruct v = (U2 ⊗ U1) * q
    % Since U1≈U3, U2≈U4 after rectification, use U1 and U2
    U_kron = kron(U2, U1);  % (d1*d2) × r²
    v = U_kron * q;         % (d1*d2) × 1
    
    % Scale by eigenvalue
    v = v * sqrt(abs(lambda_max));
    
    % Step 5: Reshape v to get X (d1 × d2)
    X = reshape(v, [d1, d2]);
    
    % Normalize
    X = X / norm(X, 'fro');
end

function X = extract_matrix_from_tucker_3(T_tucker, max_iter, tol)
    % EXTRACT_MATRIX_FROM_TUCKER_3 Extract matrix using Projected Power Method
    % For 4th-order tensor T = X ⊗ X, extract X via projected power iteration
    %
    % Method 3 (PROJECTED POWER METHOD): 
    %   1. Rectify sign ambiguity: U1≈U3, U2≈U4
    %   2. Form G_mat (r²×r²) from core tensor G (r×r×r×r)
    %   3. Power iteration with projection to diagonal support
    %   4. Support = {(k-1)*r + k : k=1...r} (diagonal positions)
    %   5. Reconstruct v = (U2⊗U1) * q
    %   6. Reshape v to get X (d1×d2)
    %
    % Advantages:
    %   - Exploits diagonal structure: G(i,j,k,l) ≈ 0 unless i=j AND k=l
    %   - Robust to off-diagonal noise
    %   - Theoretically converges to diagonal-supported solution
    %
    % Inputs:
    %   T_tucker: Tucker tensor object
    %   max_iter: Maximum power iterations (default: 50)
    %   tol: Convergence tolerance (default: 1e-6)
    %
    % For non-square matrices: d1×d2×d1×d2 tensor
    
    if nargin < 2, max_iter = 50; end
    if nargin < 3, tol = 1e-6; end
    
    d1 = T_tucker.dims(1);
    d2 = T_tucker.dims(2);
    r = T_tucker.tucker_ranks(1);
    
    % Extract factor matrices and core
    U1 = T_tucker.U{1};  % d1 × r
    U2 = T_tucker.U{2};  % d2 × r
    U3 = T_tucker.U{3};  % d1 × r
    U4 = T_tucker.U{4};  % d2 × r
    G = T_tucker.G;      % r × r × r × r
    
    % Step 1: Rectify sign ambiguity to enforce U1≈U3, U2≈U4
    U3_rect = U3;
    U4_rect = U4;
    
    for i = 1:r
        % Align U3(:,i) with U1(:,i)
        if dot(U1(:,i), U3(:,i)) < 0
            U3_rect(:,i) = -U3(:,i);
        end
        % Align U4(:,i) with U2(:,i)
        if dot(U2(:,i), U4(:,i)) < 0
            U4_rect(:,i) = -U4(:,i);
        end
    end
    
    % Sign correction matrices
    S3 = diag(sign(diag(U3_rect' * U3)));
    S4 = diag(sign(diag(U4_rect' * U4)));
    
    % Apply sign corrections to core
    G_rect = G;
    for c = 1:r
        for d = 1:r
            G_rect(:, :, c, d) = G(:, :, c, d) * S3(c,c) * S4(d,d);
        end
    end
    
    % Step 2: Form G_mat (r² × r²) from G_rect
    G_mat = reshape(G_rect, [r*r, r*r]);
    % No symmetrization - use as-is
    
    % Step 3: Initialize on diagonal support
    % Diagonal support: positions (k-1)*r + k for k=1...r
    q = zeros(r*r, 1);
    for k = 1:r
        idx = (k-1)*r + k;
        q(idx) = 1/sqrt(r);
    end
    
    % Step 4: Projected Power Iteration
    for iter = 1:max_iter
        q_old = q;
        
        % Power iteration step
        q = G_mat * q;
        
        % Project to diagonal support: keep only (k-1)*r + k positions
        q_proj = zeros(r*r, 1);
        for k = 1:r
            idx = (k-1)*r + k;
            q_proj(idx) = q(idx);
        end
        q = q_proj;
        
        % Normalize
        q_norm = norm(q);
        if q_norm > 1e-10
            q = q / q_norm;
        else
            warning('extract_matrix_from_tucker_3: q became zero during iteration');
            break;
        end
        
        % Check convergence
        conv_err = min(norm(q - q_old), norm(q + q_old));
        if conv_err < tol
            break;
        end
    end
    
    % Step 5: Compute Rayleigh quotient (eigenvalue estimate)
    lambda_max = q' * G_mat * q;
    
    % Step 6: Reconstruct v = (U2 ⊗ U1) * q
    U_kron = kron(U2, U1);  % (d1*d2) × r²
    v = U_kron * q;         % (d1*d2) × 1
    
    % Scale by eigenvalue
    v = v * sqrt(abs(lambda_max));
    
    % Step 7: Reshape v to get X (d1 × d2)
    X = reshape(v, [d1, d2]);
    
    % Normalize
    X = X / norm(X, 'fro');
end

