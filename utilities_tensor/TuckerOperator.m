classdef TuckerOperator
    % TUCKEROPERATOR Linear operators acting on Tucker tensors
    %
    % This class provides efficient forward and adjoint operators for
    % Tucker tensors without forming the full tensor.
    %
    % For 4th-order symmetric tensors (X ⊗ X structure with A_i ⊗ A_i):
    %   Forward:  y_i = <A_i ⊗ A_i, T> where T = G ×₁ U ×₂ U ×₃ U ×₄ U
    %   Adjoint:  Computes gradients ∇_G and ∇_U from residual
    %
    % The key insight: Never form A_i ⊗ A_i or the full d^N tensor!
    % Work only with A_i (d×d matrices) and Tucker factors.
    %
    % Example:
    %   % Setup
    %   A_cells = cell(m, 1);  % Measurement matrices
    %   op = TuckerOperator(A_cells, 'order', 4, 'symmetric', true);
    %   
    %   % Forward: y = A(T)
    %   y = op.forward(T_tucker);
    %   
    %   % Adjoint: [dG, dU] = A*(residual)
    %   [dG, dU] = op.adjoint(residual, T_tucker);
    % Check List
    % G_pinv dim, order sensitive? 
    % order in forward. Can be tested by comparing with tensor_forward (small scale tests s)
    %   r=1 corner case, should treat G as scalar, can not form (1,1,1,1)
    %
    %
    %
    %% Properties
    properties
        order           % Tensor order
        A_cells         % Cell array of measurement matrices
        A_mat           % Matrix with vec(A_i) as columns (d² × m)
        m               % Number of measurements
        dims            % Dimensions of each mode
        is_symmetric    % Whether operator has symmetric structure
        operator_type   % Type: 'kronecker', 'general'
    end
    
    methods
        function obj = TuckerOperator(A_cells, varargin)
            % TUCKEROPERATOR Constructor
            %
            % Syntax:
            %   op = TuckerOperator(A_cells)
            %   op = TuckerOperator(A_cells, 'order', 4, 'symmetric', true)
            %
            % Inputs:
            %   A_cells - Cell array of measurement matrices {A₁, A₂, ..., A_m}
            %             For 4th-order: each A_i is (d×d) matrix
            %   A_mat - Measurement matrix of size (m × d^2), each row is vec(A_i)
            % Optional Name-Value Pairs:
            %   'order'      - Tensor order (default: 4)
            %   'symmetric'  - Kronecker structure A_i ⊗ A_i (default: true)
            %   'dims'       - Dimensions [d₁, d₂, ...] (auto-detected if empty)
            
            p = inputParser;
            addRequired(p, 'A_cells');
            addParameter(p, 'order', 4);
            addParameter(p, 'symmetric', false);
            addParameter(p, 'dims', []);
            addParameter(p, 'operator_type', 'kronecker');
            parse(p, A_cells, varargin{:});
            
            obj.A_cells = A_cells;
            obj.m = length(A_cells);
            obj.order = p.Results.order;
            obj.is_symmetric = p.Results.symmetric;
            
            % Auto-detect dimensions from first measurement matrix
            if isempty(p.Results.dims)
                if obj.order == 4
                    % A_i is d×d, full tensor is d×d×d×d
                    d = size(A_cells{1}, 1);
                    obj.dims = [d, d, d, d];
                % else
                %     error('Must provide dims for non-standard operator');
                end
            else
                obj.dims = p.Results.dims;
            end
            
            
            obj.operator_type = p.Results.operator_type;  % A_i ⊗ A_i structure
            % else
            %     obj.operator_type = 'general';
            % end
        end
        
        function y = forward(obj, T_tucker)
            % FORWARD Apply forward operator: y = A(T)
            %
            % For symmetric 4th-order (Kronecker structure):
            %   y_i = <A_i ⊗ A_i, T> = vec(B)' * G_mat * vec(B)
            %   where B = U' * A_i * U
            %
            % Complexity: O(m * d * r²) instead of O(m * d⁴)
            %
            % Input:
            %   T_tucker - TuckerTensor object
            %
            % Output:
            %   y - Measurement vector (m × 1)
            
            if ~isa(T_tucker, 'TuckerTensor')
                error('Input must be TuckerTensor object');
            end
            
            switch obj.operator_type
                case 'kronecker'
                    y = obj.forward_kronecker(T_tucker);
                case 'general'
                    y = obj.forward_general(T_tucker);
                otherwise
                    error('Unknown operator type');
            end
        end
        
        
        
        function y = forward_kronecker(obj, T_tucker)
            % Forward operator for Kronecker structure (4th-order)
            % y_i = <A_i ⊗ A_i, G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄>
            % Handles both symmetric (U₁=U₂=U₃=U₄) and non-symmetric cases
            
            if T_tucker.order ~= 4
            error('Kronecker operator requires 4th-order tensor');
            end
        


            G = T_tucker.G;
            U1 = T_tucker.U{1};
            U2 = T_tucker.U{2};
            U3 = T_tucker.U{3};
            U4 = T_tucker.U{4};
            [d, r] = size(U1);
            
            % Unfold core tensor: G_mat is (r² × r²)
            G = T_tucker.G;
            if isscalar(G)
                G_mat = G * ones(r*r, r*r); % Handle scalar core
            else
                G_mat = reshape(permute(G, [1,2,3,4]), [r*r, r*r]);
            end

            % Compute B₁_i = U₁' * A_i * U₂ and B₂_i = U₃' * A_i * U₄ for all measurements
            B1_mat = zeros(r*r, obj.m);  % Each column is vec(U₁' * A_i * U₂)
            B2_mat = zeros(r*r, obj.m);  % Each column is vec(U₃' * A_i * U₄)
            
            for i = 1:obj.m
            Ai = reshape(obj.A_cells{i}, [d, d]);
            B1 = U1' * Ai * U2;
            B2 = U3' * Ai * U4;
            B1_mat(:, i) = B1(:);
            B2_mat(:, i) = B2(:);
            end
            
            % Vectorized inner product: y_i = vec(B₁_i)' * G_mat * vec(B₂_i)
            y = sum((G_mat * B2_mat) .* B1_mat, 1)';
        end

        function Grad_F = get_proj_grad_kronecker(obj, T_tucker, y0, y)
            % GET_PROJ_GRAD_KRONECKER Compute Riemannian gradient on Tucker manifold
            % Computes the tangent space projection of the Euclidean gradient
            % Based on projection formula from TuckerTensor_projection.md
            %
            % For measurement model: y_i = <A_i ⊗ A_i, T>
            % Euclidean gradient: ∇F = Σ_i (y_i - y_i^true) * (A_i ⊗ A_i)
            % Riemannian gradient: Project ∇F onto tangent space of Tucker manifold
            %
            % Inputs:
            %   T_tucker - Current TuckerTensor point
            %   y0       - Computed measurements (forward of T_tucker)
            %   y        - True measurements
            %
            % Output:
            %   Grad_F   - TuckerTensor with Riemannian gradient on tangent space
            %              .G: Core gradient (in tangent space)
            %              .U{k}: Factor gradients (orthogonal to U{k})
            %              .Up{k}: Orthogonal complement bases
            %   has tested: sum then project vs project then sum give same result - find bug on tensor order
            
            if T_tucker.order ~= 4
                error('Kronecker operator requires 4th-order tensor');
            end
            
            % Compute residual
            residual = y0 - y;
            
            % Extract components
            G = T_tucker.G;
            U1 = T_tucker.U{1};
            U2 = T_tucker.U{2};
            U3 = T_tucker.U{3};
            U4 = T_tucker.U{4};
            [d, r] = size(U1);
            
            % Initialize gradient as Tucker tensor
            Grad_F = TuckerTensor(T_tucker.dims, T_tucker.tucker_ranks, ...
                                  'symmetric', T_tucker.is_symmetric, ...
                                  'init_method', 'zeros');
            
            % Copy U from T_tucker (Up will be computed below)
            for k = 1:4
                Grad_F.U{k} = T_tucker.U{k};
            end
            
            %% Compute Core Gradient: dG
            % Core variation from projection formula: A ×_{k=1}^d P_{U_k}
            % For Kronecker: Σ_i residual_i * (U₁' ⊗ U₂' ⊗ U₃' ⊗ U₄') * (A_i ⊗ A_i)
            % Efficiently: dG[j1,j2,j3,j4] = Σ_i residual_i * (U₁'*A_i*U₂)[j1,j2] * (U₃'*A_i*U₄)[j3,j4]
            dG = zeros(size(G));
            for i = 1:obj.m
                Ai = obj.A_cells{i};
                B1 = U1' * Ai * U2;  % r × r
                B2 = U3' * Ai * U4;  % r × r
                % Outer product: B1 ⊗ B2 as 4D tensor
                dG = dG + residual(i) * reshape(B1(:) * B2(:)', [r, r, r, r]);
            end
            Grad_F.G = dG;
            for k=1:4
                Grad_F.U{k} = T_tucker.U{k};
            end
            
            %% Compute Factor Gradients: dU{k}
            % Based on projection formula:
            % P_⊥_{U_k} * (A ×_{j≠k} U_j^T)^(k) * G^(k)†
            %
            % For A_i ⊗ A_i structure and mode k:
            % dU_k = (I - U_k*U_k') * [Σ_i residual_i * gradient_term_k]
            
            % Precompute G mode unfoldings and pseudoinverses for efficiency
            G_unfold = cell(4, 1);
            G_pinv = cell(4, 1);
            
            % Special case: r=1, G is scalar (or collapsed to scalar)
            if r == 1
                % For r=1, all mode unfoldings are the same scalar
                G_scalar = G(1);  % Extract scalar value
                for k = 1:4
                    G_unfold{k} = G_scalar;  % 1 × 1 matrix
                    if abs(G_scalar) > 1e-14
                        G_pinv{k} = 1 / G_scalar;  % Scalar pseudoinverse
                    else
                        warning('Core tensor G is near zero scalar; pseudoinverse may be unstable.');
                    end
                end
            else
                % Normal case: r > 1
                for k = 1:4
                    % Mode-k unfolding: G(k) is r × r³
                    perm = [k, setdiff(1:4, k)];
                    G_perm = permute(G, perm);
                    G_unfold{k} = reshape(G_perm, r, []);
                    G_pinv{k} = pinv(G_unfold{k});
                end
            end
            
            %% Mode 1: dUp₁
            % Gradient term: Σ_i residual_i * A_i ×₂ U₂^T ×₃ U₃^T ×₄ U₄^T
            % Result is d × r³ matrix after mode-1 unfolding
            grad_term_1 = zeros(d, r^3);
            for i = 1:obj.m
                Ai = obj.A_cells{i};
                % For A_i ⊗ A_i, mode-1 unfolding gives contribution:
                % Ai ⊗ (U₂^T ⊗ U₃^T ⊗ U₄^T applied to second copy)
                % Efficiently compute for each component
                
                % Build (A_i ⊗ A_i) ×₂ U₂^T ×₃ U₃^T ×₄ U₄^T in mode-1 unfolding
                B1 =  Ai * U2;  % d × r
                B2 = U3' * Ai * U4;  % r × r
                grad_term_1 = grad_term_1 + residual(i)  * reshape( (reshape(B1, [d*r,1]) * reshape(B2, [1, r*r])), [d , r^3]);
                % now the vec order is 4 3 2, correct
            end
            % Project to tangent space: (I - U₁*U₁') * grad_term_1 * G_pinv{1}
            P_perp_1 = grad_term_1  - U1 * (U1'*grad_term_1);
            Grad_F.Up{1} = P_perp_1  * G_pinv{1}; % careful match dimensions, I think we don't need transpose
            % size of Grad_F.U{1}: d × r
            % For symmetric case, use symmetry
            if T_tucker.is_symmetric
                % All modes have same contribution (with symmetry)
                for k = 2:4
                    Grad_F.Up{k} = Grad_F.Up{1};
                end
            else
                %% Mode 2: dUp₂
                % Gradient term: Σ_i residual_i * A_i^T ×₁ U₁^T ×₃ U₃^T ×₄ U₄^T
                grad_term_2 = zeros(d, r^3);
                for i = 1:obj.m
                    Ai = obj.A_cells{i};
                    % Build (A_i ⊗ A_i) ×₁ U₁^T ×₃ U₃^T ×₄ U₄^T in mode-2 unfolding
                    B1 = Ai' * U1;  % d × r
                    B2 = U3' * Ai * U4;  % r × r
                    grad_term_2 = grad_term_2 + residual(i) * reshape((reshape(B1, [d*r, 1]) * reshape(B2, [1, r*r])), [d, r^3]);
                end
                % now the vec order is 4 3 1, correct
                % Project to tangent space: (I - U₂*U₂') * grad_term_2 * G_pinv{2}
                P_perp_2 = grad_term_2 - U2 * (U2' * grad_term_2);
                Grad_F.Up{2} = P_perp_2 * G_pinv{2};  % No transpose: (d×r³) × (r³×r) = d×r
                
                %% Mode 3: dUp₃
                % Gradient term: Σ_i residual_i * A_i ×₁ U₁^T ×₂ U₂^T ×₄ U₄^T
                grad_term_3 = zeros(d, r, r, r);
                for i = 1:obj.m
                    Ai = obj.A_cells{i};
                    % Build (A_i ⊗ A_i) ×₁ U₁^T ×₂ U₂^T ×₄ U₄^T in mode-3 unfolding
                    B1 = Ai * U4;  % d × r
                    B2 = U1' * Ai * U2;  % r × r
                    grad_term_3 = grad_term_3 + residual(i) * reshape((reshape(B1, [d*r, 1]) * reshape(B2, [1, r*r])), [d, r, r, r]);
                    % now the vec order is 3 4 1 2, it shoud be permuted to 3 1 2 4
                end
                grad_term_3 = permute(grad_term_3, [1, 3, 4, 2]);
                grad_term_3 = reshape(grad_term_3, [d, r^3]);
                % Project to tangent space: (I - U₃*U₃') * grad_term_3 * G_pinv{3}
                P_perp_3 = grad_term_3 - U3 * (U3' * grad_term_3);
                Grad_F.Up{3} = P_perp_3 * G_pinv{3};  % No transpose: (d×r³) × (r³×r) = d×r
                
                %% Mode 4: dUp₄
                % Gradient term: Σ_i residual_i * A_i^T ×₁ U₁^T ×₂ U₂^T ×₃ U₃^T
                grad_term_4 = zeros(d, r ,r,r);
                for i = 1:obj.m
                    Ai = obj.A_cells{i};
                    % Build (A_i ⊗ A_i) ×₁ U₁^T ×₂ U₂^T ×₃ U₃^T in mode-4 unfolding
                    B1 = Ai' * U3;  % d × r
                    B2 = U1' * Ai * U2;  % r × r
                    grad_term_4 = grad_term_4 + residual(i) * reshape((reshape(B1, [d*r, 1]) * reshape(B2, [1, r*r])), [d, r , r, r]);
                    % now the vec order is 4 3 1 2, it shoud be permuted to 4 1 2 3
                end
                grad_term_4 = permute(grad_term_4, [1 , 3,4,2]);
                grad_term_4 = reshape(grad_term_4, [d, r^3]);
                % Project to tangent space: (I - U₄*U₄') * grad_term_4 * G_pinv{4}
                P_perp_4 = grad_term_4 - U4 * (U4' * grad_term_4);
                Grad_F.Up{4} = P_perp_4 * G_pinv{4};  % No transpose: (d×r³) × (r³×r) = d×r
            end
            
        end
        
        
        
        function y = forward_general(obj, T_tucker)
            % FORWARD_GENERAL General forward operator using full tensor
            % Reconstructs the full tensor and computes measurements
            %
            % WARNING: This is memory intensive! Use only for small tensors.
            %
            % y_i = <A_i ⊗ A_i, T>
            
            % Reconstruct full tensor
            T_full = T_tucker.full();
            
            % Compute measurements using vectorization
            y = zeros(obj.m, 1);
            
            if obj.order == 4
                % For 4th-order: y_i = <A_i ⊗ A_i, T>
                % Vectorize: y_i = vec(A_i)' * (A_i ⊗ I ⊗ I ⊗ I + ...) * vec(T)
                % More efficiently: reshape and use tensor contractions
                d = obj.dims(1);
                
                % Reshape T_full to matrix form for vectorized operations
                % T_full is d×d×d×d, reshape to (d²×d²)
               
                A_tensor = zeros(obj.m, d*d*d*d);
                for i = 1:obj.m
                    Ai = reshape(obj.A_cells{i}, [d, d]);
                    % Fourth-order tensor A_i ⊗ A_i
                    AiAi = reshape(Ai, d^2, 1) * reshape(Ai, 1, d^2);  % d^2 x d^2
                    A_tensor(i, :) = AiAi(:)';  % Flatten and store
                end
                y = A_tensor * reshape(T_full, [d*d*d*d, 1]);
            else
                error('General forward only supports 4th-order tensors');
            end
        end
        
        function [dG, dU] = adjoint_general(obj, z, T_tucker)
            % General adjoint operator (not yet implemented)
            error('General adjoint operator not yet implemented');
        end
        
        function display(obj)
            % Display operator information
            fprintf('TuckerOperator:\n');
            fprintf('  Type: %s\n', obj.operator_type);
            fprintf('  Order: %d\n', obj.order);
            fprintf('  Measurements: %d\n', obj.m);
            fprintf('  Dimensions: [%s]\n', num2str(obj.dims));
            fprintf('  Symmetric: %s\n', mat2str(obj.is_symmetric));
        end
    end
end
