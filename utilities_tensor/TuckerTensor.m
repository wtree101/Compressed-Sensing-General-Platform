classdef TuckerTensor
    % TUCKERTENSOR General Tucker tensor decomposition class
    % 
    % A Tucker tensor T is represented as:
    %   T = G ×₁ U₁ ×₂ U₂ ×₃ ... ×_N U_N
    % where:
    %   G is the core tensor of size (r₁ × r₂ × ... × r_N)
    %   U_i are factor matrices of size (d_i × r_i)
    %
    % This class provides:
    %   - Flexible constructor for any order and Tucker ranks
    %   - Efficient operations without forming full tensor
    %   - Support for tied factors (symmetric tensors)
    %   - Forward/adjoint operators for optimization
    %
    % Example:
    %   % Create 4th-order Tucker tensor: d=20, r=5
    %   T = TuckerTensor([20, 20, 20, 20], [5, 5, 5, 5]);
    %   
    %   % Symmetric tensor (tied factors)
    %   T_sym = TuckerTensor([20, 20, 20, 20], 5, 'symmetric', true);
    %   
    % Checklist
    % whether the result tensor would be symmetric (has the same U for all modes). I guess yes for this model. Then the result could be easily extracted.
    %
    %
    %
    %

    %% Properties
    properties
        order           % Number of modes (tensor order)
        dims            % Dimensions [d₁, d₂, ..., d_N]
        tucker_ranks    % Tucker ranks [r₁, r₂, ..., r_N]
        G               % Core tensor (r₁ × r₂ × ... × r_N)
        U               % Cell array of factor matrices {U₁, U₂, ..., U_N}
        Up              % Cell array of the orthogonal components of factor matrices on the tangent space {Up₁, Up₂, ..., Up_N}, may be empty
        is_symmetric    % Whether all factors are tied (U₁ = U₂ = ... = U_N)
        debug           % Debug mode: enables verbose fprintf output (default: false)
    end
    
    methods
        function obj = TuckerTensor(dims, tucker_ranks, varargin)
            % TUCKERTENSOR Constructor
            %
            % Syntax:
            %   T = TuckerTensor(dims, tucker_ranks)
            %   T = TuckerTensor(dims, r, 'symmetric', true)
            %   T = TuckerTensor(dims, tucker_ranks, 'G', G_init, 'U', U_init)
            %
            % Inputs:
            %   dims         - Vector of mode dimensions [d₁, d₂, ..., d_N]
            %   tucker_ranks - Vector of Tucker ranks [r₁, r₂, ..., r_N]
            %                  OR scalar r (all ranks equal)
            %
            % Optional Name-Value Pairs:
            %   'symmetric'  - Boolean, tie all factors (default: false)
            %   'G'          - Initial core tensor
            %   'U'          - Initial factor matrices (cell array)
            %   'init_method' - 'zeros', 'random', 'orthogonal', 'spectral' (default: 'zeros')
            %   'operator'   - Operator struct (required for 'spectral' method)
            %                  .A_star: Adjoint operator @(y) reshape(A'*y, [d1,d2])
            %   'y'          - Measurement vector (required for 'spectral' method)
            %   'm'          - Number of measurements (required for 'spectral' method)
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'dims');
            addRequired(p, 'tucker_ranks');
            addParameter(p, 'symmetric', false);
            addParameter(p, 'G', []);
            addParameter(p, 'U', []);
            addParameter(p, 'init_method', 'zeros');
            addParameter(p, 'operator', []);
            addParameter(p, 'y', []);
            addParameter(p, 'm', []);
            addParameter(p, 'debug', false);
            parse(p, dims, tucker_ranks, varargin{:});
            
            % Set dimensions
            obj.order = length(dims);
            obj.dims = dims(:)';  % Row vector
            
            % Handle scalar tucker_ranks (all modes same rank)
            if isscalar(tucker_ranks)
                obj.tucker_ranks = tucker_ranks * ones(1, obj.order);
            else
                obj.tucker_ranks = tucker_ranks(:)';  % Row vector
            end
            
            % Validate dimensions
            
            if length(obj.dims) ~= obj.order
                error('dims must have length equal to tensor order');
            end
            if length(obj.tucker_ranks) ~= obj.order
                error('tucker_ranks must have length equal to tensor order');
            end
            % if obj.tucker_ranks(1) == 1
            %     disp('Warning: Tucker rank of 1 may lead to degenerate core tensor. Careful!');
            % end
            
            % Set debug mode
            obj.debug = p.Results.debug;
            
            % Set symmetry flag
            obj.is_symmetric = p.Results.symmetric;
            if obj.is_symmetric
                % For symmetric tensors, all dims and ranks must match
                if length(unique(obj.dims)) > 1
                    error('Symmetric tensor requires all dims to be equal');
                end
                if length(unique(obj.tucker_ranks)) > 1
                    warning('Symmetric tensor: using first rank for all modes');
                    obj.tucker_ranks = obj.tucker_ranks(1) * ones(1, obj.order);
                end
            end
            
            % Initialize core tensor G
            if ~isempty(p.Results.G)
                obj.G = p.Results.G;
                % Validate size
                expected_size = obj.tucker_ranks;
                if ~isequal(size(obj.G), expected_size)
                    error('Core tensor G has wrong size');
                end
            else
                % Initialize based on method
                if strcmpi(p.Results.init_method, 'spectral')
                    % Spectral initialization needs special handling
                    % Core will be initialized after factors
                    obj.G = obj.initialize_core('zeros');
                else
                    obj.G = obj.initialize_core(p.Results.init_method);
                end
            end
            
            % Initialize factor matrices U
            if ~isempty(p.Results.U)
                obj.U = p.Results.U;
                % Validate
                if length(obj.U) ~= obj.order
                    error('U must be cell array of length %d', obj.order);
                end
            else
                obj.U = cell(1, obj.order);
                
                % Handle spectral initialization
                if strcmpi(p.Results.init_method, 'spectral')
                    [obj.U, obj.G] = obj.initialize_spectral(p.Results.operator, p.Results.y, p.Results.m, ...
                                                             'symmetric', obj.is_symmetric);
                else
                    obj.U{1} = obj.initialize_factor(1, p.Results.init_method);
                    for i = 2:obj.order
                        obj.U{i} = obj.U{1};
                    end
                end
            end
            
            % Initialize Up (orthogonal complement) - initially empty
            obj.Up = cell(1, obj.order);
            for i = 1:obj.order
                obj.Up{i} = [];
            end
        end
        
        function G_init = initialize_core(obj, method)
            % Initialize core tensor based on method
            switch lower(method)
                case 'zeros'
                    G_init = zeros(obj.tucker_ranks);
                case 'random'
                    G_init = randn(obj.tucker_ranks) * 0.1;
                case 'diagonal'
                    % Identity-like initialization for diagonal elements
                    G_init = zeros(obj.tucker_ranks);
                    if obj.order == 4 && length(unique(obj.tucker_ranks)) == 1
                        r = obj.tucker_ranks(1);
                        for i = 1:min(r, obj.tucker_ranks(1))
                            G_init(i, i, i, i) = 1.0;
                        end
                    end
                case 'ones'
                    G_init = ones(obj.tucker_ranks)*0.1;
                otherwise
                    G_init = zeros(obj.tucker_ranks);
            end
        end
        
        function U_init = initialize_factor(obj, mode, method)
            % Initialize factor matrix for given mode
            d = obj.dims(mode);
            r = obj.tucker_ranks(mode);
            
            switch lower(method)
                case 'zeros'
                    U_init = zeros(d, r);
                case {'random', 'ones'}
                    U_init = orth(randn(d, r));
                case 'spectral'
                    % Spectral initialization is handled separately in initialize_spectral
                    U_init = orth(randn(d, r));
                otherwise
                    U_init = zeros(d, r);
            end
        end
        
        function [U_cell, G_init] = initialize_spectral(obj, operator, y, m, varargin)
            % INITIALIZE_SPECTRAL Spectral initialization for Tucker tensor
            %
            % This method performs spectral initialization by directly forming
            % T = sum_i y_i * (Ai ⊗ Ai) using operator.A_cells, then applying HOSVD.
            %
            % Inputs:
            %   operator - Struct with .A_cells: Cell array of measurement matrices {A₁, A₂, ..., A_m}
            %   y        - Measurement vector
            %   m        - Number of measurements
            %   varargin - Optional name-value pairs:
            %              'pre_func' - Preprocessing function handle applied to y before forming T
            %                           Default: @(y) y (identity, no preprocessing)
            %                           Example: @(y) set_zero_outside_range(y)
            %              'symmetric' - Boolean, use same factor for all modes (default: true)
            %                            If false, uses all extracted factors from HOSVD
            %
            % Outputs:
            %   U_cell   - Cell array of factor matrices
            %   G_init   - Core tensor
            %
            % Steps:
            %   0. Apply pre_func to y (if provided)
            %   1. Form T = sum_i y_i * (Ai ⊗ Ai) directly as (d×d×d×d) array
            %   2. Perform HOSVD on T to get U{1}...U{N}
            %   3. Compute core G = T ×₁ U₁' ×₂ U₂' ×₃ U₃' ×₄ U₄'
            %
            % Memory usage: d^4 (avoids forming md^4 A_tensor array)
            % Time complexity: O(m*d^4 + d^5)
            %
            % Example:
            %   [U_cell, G] = obj.initialize_spectral(op, y, m, 'pre_func', @set_zero_outside_range);
            %   [U_cell, G] = obj.initialize_spectral(op, y, m, 'symmetric', false);
            
            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'pre_func', @(y) y, @(f) isa(f, 'function_handle'));
            addParameter(p, 'symmetric', obj.is_symmetric, @(x) islogical(x) || isnumeric(x));
            parse(p, varargin{:});
            pre_func = p.Results.pre_func;
            use_symmetric = p.Results.symmetric;
            
            % Validate inputs
            if isempty(operator) || ~isfield(operator, 'A_cells')
                error('For spectral initialization, operator with A_cells field is required');
            end
            if isempty(y)
                error('For spectral initialization, measurement vector y is required');
            end
            if isempty(m)
                m = length(y);
            end
            
            % Check tensor order (spectral init typically for 4th-order)
            if obj.order ~= 4
                warning('Spectral initialization optimized for 4th-order tensors');
            end
            
            % Get first dimension (assume symmetric case)
            d = obj.dims(1);
            tucker_rank = obj.tucker_ranks(1);
            
            % Memory tracking
            if obj.debug
                fprintf('[Spectral Init] Starting initialization: d=%d, r=%d, m=%d\n', d, tucker_rank, m);
            end
            
            % Step 0: Apply preprocessing function to measurements
            y_processed = pre_func(y);
            n_kept = sum(y_processed ~= 0);
            if obj.debug && n_kept < m
                fprintf('[Spectral Init] Preprocessing applied: %d/%d measurements kept (%.1f%%)\n', ...
                        n_kept, m, n_kept/m*100);
            end
            
            % Step 1: Form T = sum_i y_i * (Ai ⊗ Ai) using TuckerOperator
            if obj.debug
                fprintf('[Spectral Init] Forming tensor T = sum_i y_i * (Ai ⊗ Ai)...\n');
            end
            
            % Check if operator is already a TuckerOperator, otherwise create one
            if isa(operator, 'TuckerOperator')
                tucker_op = operator;
                if obj.debug
                    fprintf('[Spectral Init] Using existing TuckerOperator\n');
                end
            else
                % Create TuckerOperator from A_cells
                tucker_op = TuckerOperator(operator.A_cells, 'order', obj.order, ...
                                           'symmetric', false, 'operator_type', 'kronecker');
                if obj.debug
                    fprintf('[Spectral Init] Created new TuckerOperator from A_cells\n');
                end
            end
            
            % Scale measurements and apply Kronecker adjoint
            y_scaled = y_processed / sqrt(m);
            T = tucker_op.kronecker_adjoint(y_scaled);
            
            mem_T = numel(T) * 8 / 1024;  % KB
            if obj.debug
                fprintf('[Spectral Init] Memory: T (%dx%dx%dx%d) = %.2f KB\n', ...
                        d, d, d, d, mem_T);
                fprintf('[Spectral Init] Tensor T formed: norm=%.6f\n', norm(T(:)));
            end
            
            % Step 2: Apply HOSVD using new HOSVD_with_factors function
            if obj.debug
                fprintf('[Spectral Init] Performing HOSVD...\n');
            end
            
            rank_vec = tucker_rank * ones(1, obj.order);
            
            % Use HOSVD_with_factors to get both projected tensor and factor matrices
            % This function internally does: tensor_to_mat + svds for each mode + projection
            % Returns: T_hosvd = mul(...mul(mul(T, 1, U1*U1'), 2, U2*U2')..., 4, U4*U4')
            %          U_cells = {U1, U2, U3, U4}
            [T_hosvd, U_cells] = HOSVD_with_factors(T, rank_vec);
            mem_T_hosvd = numel(T_hosvd) * 8 / 1024;  % KB
            if obj.debug
                fprintf('[Spectral Init] Memory: T_hosvd (via HOSVD) (%dx%dx%dx%d) = %.2f KB\n', ...
                        size(T_hosvd,1), size(T_hosvd,2), size(T_hosvd,3), size(T_hosvd,4), mem_T_hosvd);
                
                fprintf('[Spectral Init] HOSVD complete: extracted %d factor matrices\n', length(U_cells));
            end
            
            % Handle symmetric vs non-symmetric initialization
            if use_symmetric
                % For symmetric tensor, use first factor for all modes
                U_factor = U_cells{1};
                mem_U_factor = numel(U_factor) * 8 / 1024;  % KB
                if obj.debug
                    fprintf('[Spectral Init] Memory: U_factor (%dx%d) = %.2f KB\n', ...
                            size(U_factor,1), size(U_factor,2), mem_U_factor);
                end
                
                U_cell = cell(1, obj.order);
                for i = 1:obj.order
                    U_cell{i} = U_factor;
                end
                mem_U_cell = obj.order * mem_U_factor;
                if obj.debug
                    fprintf('[Spectral Init] Memory: U_cell (%d factors, SYMMETRIC) = %.2f KB\n', obj.order, mem_U_cell);
                end
            else
                % For non-symmetric tensor, use all extracted factors
                U_cell = U_cells;
                mem_U_cell = 0;
                if obj.debug
                    fprintf('[Spectral Init] Using non-symmetric factors from HOSVD:\n');
                end
                for i = 1:length(U_cell)
                    mem_i = numel(U_cell{i}) * 8 / 1024;  % KB
                    mem_U_cell = mem_U_cell + mem_i;
                    if obj.debug
                        fprintf('[Spectral Init]   U_cell{%d}: %dx%d = %.2f KB\n', ...
                                i, size(U_cell{i},1), size(U_cell{i},2), mem_i);
                    end
                end
                if obj.debug
                    fprintf('[Spectral Init] Memory: U_cell (%d factors, NON-SYMMETRIC) = %.2f KB\n', ...
                            length(U_cell), mem_U_cell);
                end
            end
            
            % Step 3: Compute core tensor by projecting T onto U factors
            % Core G = T ×₁ U₁' ×₂ U₂' ×₃ U₃' ×₄ U₄'
            if obj.debug
                fprintf('[Spectral Init] Computing core tensor via tensor_mode_product...\n');
            end
            
            G_init = T;
            if use_symmetric
                % For symmetric case, use same factor for all modes
                U_factor = U_cell{1};
                if obj.debug
                    fprintf('[Spectral Init]   Projecting with same factor on all %d modes\n', obj.order);
                end
                for mode = 1:obj.order
                    G_init = tensor_mode_product(G_init, U_factor', mode);
                end
            else
                % For non-symmetric case, use corresponding factor for each mode
                if obj.debug
                    fprintf('[Spectral Init]   Projecting with distinct factors on each mode:\n');
                end
                for mode = 1:obj.order
                    if obj.debug
                        fprintf('[Spectral Init]     Mode %d: projecting with U{%d} (%dx%d)\n', ...
                                mode, mode, size(U_cell{mode},1), size(U_cell{mode},2));
                    end
                    G_init = tensor_mode_product(G_init, U_cell{mode}', mode);
                end
            end
            
            mem_G_init = numel(G_init) * 8 / 1024;  % KB
            if obj.debug
                fprintf('[Spectral Init] Memory: G_init (%s) = %.2f KB\n', ...
                        mat2str(size(G_init)), mem_G_init);
                
                % Total memory usage
                mem_total = mem_T + mem_T_hosvd + mem_U_cell + mem_G_init;
                fprintf('[Spectral Init] Total memory usage: %.2f KB (%.2f MB)\n', ...
                        mem_total, mem_total / 1024);
                
                % Memory comparison with alternative approaches
                mem_md4 = m * d^4 * 8 / 1024 / 1024;  % MB (if forming full A_tensor)
                mem_d4 = d^4 * 8 / 1024 / 1024;  % MB (current approach)
                fprintf('[Spectral Init] Memory saved vs md^4 approach: %.2f MB vs %.2f MB (%.1fx reduction)\n', ...
                        mem_d4, mem_md4, mem_md4 / max(mem_d4, 1e-6));
                
                % Initialization summary
                if use_symmetric
                    fprintf('[Spectral Init] === Spectral initialization complete (SYMMETRIC) ===\n');
                else
                    fprintf('[Spectral Init] === Spectral initialization complete (NON-SYMMETRIC) ===\n');
                end
            end
        end
        
        function T_full = full(obj)
            % FULL Reconstruct full tensor from Tucker decomposition
            % T = G ×₁ U₁ ×₂ U₂ ×₃ ... ×_N U_N
            %
            % WARNING: This can be very memory intensive!
            % Only use for small tensors or debugging.
            
            T_full = obj.G;
            % If core is scalar, expand to tensor product of all U
            if isscalar(T_full)
                % Tensor product of all factor matrices
                T_full = obj.U{1};
                for i = 2:obj.order
                    T_full = kron(T_full, obj.U{i});
                end
                % Reshape to full tensor size
                T_full = reshape(T_full, obj.dims) * obj.G;
            else
                for i = 1:obj.order
                    T_full = tensor_mode_product(T_full, obj.U{i}, i);
                end
            end
        end
        
        function R_mat = retraction(obj, Grad_F, eta)
            % RETRACTION Retraction operation on Tucker manifold
            % Implements retraction using tangent space projection and SVD truncation
            %
            % Inputs:
            %   Grad_F - TuckerTensor gradient with U and Up components
            %   eta    - Step size
            %
            % Output:
            %   R_mat  - Retracted TuckerTensor on the manifold
            
            % Initialize return tensor
            R_mat = TuckerTensor(obj.dims, obj.tucker_ranks, ...
                                 'symmetric', obj.is_symmetric, ...
                                 'init_method', 'zeros');
            
            % Build extended core tensor of size (2r₁ × 2r₂ × ... × 2r_N)
            extended_size = 2 * obj.tucker_ranks;
            Core = zeros(extended_size);
            
            % Core(1:r₁, 1:r₂, ..., 1:r_N) = G - η·Grad_F.G
            index_ranges = cell(1, obj.order);
            for i = 1:obj.order
                index_ranges{i} = 1:obj.tucker_ranks(i);
            end
            Core(index_ranges{:}) = obj.G - Grad_F.G * eta;
            
            % Core(1:r₁, ..., r_i+1:2r_i, ..., 1:r_N) = -eta G(...) for each mode i
            for i = 1:obj.order
                index_temp = cell(1, obj.order);
                for j = 1:obj.order
                    if j == i
                        index_temp{j} = (obj.tucker_ranks(j) + 1):(2 * obj.tucker_ranks(j));
                    else
                        index_temp{j} = 1:obj.tucker_ranks(j);
                    end
                end
                Core(index_temp{:}) = -eta * obj.G;
            end
            
            % % Build extended factor matrices and perform QR decomposition
            % Qlist = cell(1, obj.order);
            % for j = 1:obj.order
            %     % Concatenate U and Up: [U, Up] where Up is the gradient direction
            %     UUp = [obj.U{j}, Grad_F.Up{j}];
            %     [Qu, Ru] = qr(UUp, 0);
            %     Qlist{j} = Qu;
            %     % Apply Ru to core tensor via mode-j product
            %     Core = tensor_mode_product(Core, Ru, j);
            % end
            
            % % Truncate back to rank r via SVD on each mode
            % B = Core;
            % for j = 1:obj.order  %when r = 1, have to be careful and j from 1 to end, otherwise collapse
            %     % Mode-j unfolding
            %     mat = mode_k_unfold(B, j, extended_size);
                
            %     % SVD and truncate to rank r_j
            %     [u, ~, ~] = svd(mat, 'econ');
            %     r_j = obj.tucker_ranks(j);
                
            %     % Extract first r_j components
            %     R_mat.U{j} = Qlist{j} * u(:, 1:r_j);
                
            %     % Update core by contracting with truncated left singular vectors
            %     B = tensor_mode_product(B, u(:, 1:r_j)', j);
            % end
            
            % % Final core tensor
            % R_mat.G = B;
            
            % % Copy symmetry if applicable
            % if obj.is_symmetric
            %     for j = 2:obj.order
            %         R_mat.U{j} = R_mat.U{1};
            %     end
            % end

            %% Alternative retraction for symmetric tensors -- debugging

            Qlist = cell(1, obj.order);
            j = 1;
            UUp = [obj.U{j}, Grad_F.Up{j}];
            [Qu, Ru] = qr(UUp, 0);
            Qlist{j} = Qu;
            % Apply Ru to core tensor via mode-j product
            Core = tensor_mode_product(Core, Ru, j);

            if obj.is_symmetric
                for j = 2:obj.order
                    Qlist{j} = Qlist{1};
                    Core = tensor_mode_product(Core, Ru, j);
                end
                
            else
                for j = 2:obj.order
                    UUp = [obj.U{j}, Grad_F.Up{j}];
                    [Qu, Ru] = qr(UUp, 0);
                    Qlist{j} = Qu;
                    % Apply Ru to core tensor via mode-j product
                    Core = tensor_mode_product(Core, Ru, j);
                end
            end

            B = Core;
            j = 1;
            mat = mode_k_unfold(B, j, extended_size);
            % SVD and truncate to rank r_j
            [u, ~, ~] = svd(mat, 'econ');
            r_j = obj.tucker_ranks(j);
            
            % Extract first r_j components
            R_mat.U{j} = Qlist{j} * u(:, 1:r_j);
            
            % Update core by contracting with truncated left singular vectors
            B = tensor_mode_product(B, u(:, 1:r_j)', j);

            if obj.is_symmetric
                for j = 2:obj.order
                    R_mat.U{j} = R_mat.U{1};
                    B = tensor_mode_product(B, u(:, 1:r_j)', j);
                end
                
            else
                for j = 2:obj.order
                    mat = mode_k_unfold(B, j, extended_size);
                    
                    % SVD and truncate to rank r_j
                    [u, ~, ~] = svd(mat, 'econ');
                    r_j = obj.tucker_ranks(j);
                    
                    % Extract first r_j components
                    R_mat.U{j} = Qlist{j} * u(:, 1:r_j);
                    
                    % Update core by contracting with truncated left singular vectors
                    B = tensor_mode_product(B, u(:, 1:r_j)', j);
                end
            end
            % Final core tensor
            R_mat.G = B;

          
        end
        
        function mem = memory_usage(obj)
            % MEMORY_USAGE Estimate memory usage in bytes
            core_mem = numel(obj.G) * 8;  % 8 bytes per double
            factor_mem = 0;
            for i = 1:obj.order
                factor_mem = factor_mem + numel(obj.U{i}) * 8;
            end
            mem = core_mem + factor_mem;
        end
        
        function display(obj)
            % DISPLAY Custom display method
            fprintf('TuckerTensor: Order-%d\n', obj.order);
            fprintf('  Dimensions: [%s]\n', num2str(obj.dims));
            fprintf('  Tucker ranks: [%s]\n', num2str(obj.tucker_ranks));
            fprintf('  Symmetric: %s\n', mat2str(obj.is_symmetric));
            fprintf('  Core size: %s\n', mat2str(size(obj.G)));
            fprintf('  Memory: %.2f KB\n', obj.memory_usage() / 1024);
            if ~obj.is_symmetric
                fprintf('  Factor matrices:\n');
                for i = 1:obj.order
                    fprintf('    U{%d}: %dx%d\n', i, size(obj.U{i}, 1), size(obj.U{i}, 2));
                end
            else
                fprintf('  Factor matrix (shared): %dx%d\n', ...
                        size(obj.U{1}, 1), size(obj.U{1}, 2));
            end
        end
    end
end

function T_out = tensor_mode_product(T, M, mode)
    % TENSOR_MODE_PRODUCT n-mode product of tensor T with matrix M
    % T_out = T ×_mode M
    
    sz = size(T);
    
    % Handle scalar/low-dimensional tensors
    % When all dimensions are 1, MATLAB drops trailing dimensions
    if isscalar(T)
        % T is scalar, treat as 1×1×...×1 tensor
       warning('TuckerTensor:ScalarTensor', 'Tensor is scalar, treating as 1x1x...x1 tensor');
    end
    
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

function T_mat = mode_k_unfold(T, k, dims)
    % MODE_K_UNFOLD Mode-k unfolding (matricization) of tensor T
    % Returns matrix T_mat of size (dims(k) × prod(dims([1:k-1, k+1:end])))
    %
    % Input:
    %   T    - Tensor to unfold
    %   k    - Mode for unfolding
    %   dims - Dimensions vector
    %
    % Output:
    %   T_mat - Matricized tensor
    
    n_modes = length(dims);
    
    % Permute so mode k is first
    perm = [k, 1:k-1, k+1:n_modes];
    T_perm = permute(T, perm);
    
    % Reshape to matrix: mode-k fibers as rows
    T_mat = reshape(T_perm, dims(k), []);
end

