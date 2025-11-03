%% Test: Gradient Projection Order - Project-then-Sum vs Sum-then-Project
% 
% This test verifies that the Riemannian gradient computation is correct by
% comparing two equivalent approaches:
%
% Method 1 (Current implementation): Project each measurement's contribution, then sum
%   Grad_F = Σ_i residual_i * Project(A_i ⊗ A_i)
%
% Method 2 (Reference): Compute full Euclidean gradient, then project
%   Grad_F = Project(Σ_i residual_i * A_i ⊗ A_i)
%
% These should give identical results due to linearity of projection.

clear; clc;

fprintf('=== Testing Gradient Projection Order ===\n\n');

%% Test Parameters
d = 8;    % Dimension (keep small for full tensor)
r = 3;    % Tucker rank
m = 50;   % Number of measurements

fprintf('Test setup: d=%d, r=%d, m=%d\n\n', d, r, m);

%% Create Measurement Matrices
fprintf('Creating %d measurement matrices...\n', m);
A_cells = cell(m, 1);
A_mat = zeros(d*d, m);
for i = 1:m
    Ai = randn(d, d);
    Ai = (Ai + Ai') / 2;  % Make symmetric
    A_cells{i} = Ai;
    A_mat(:, i) = Ai(:);
end

%% Create Tucker Tensor and Measurements
fprintf('Creating Tucker tensor...\n');
T_tucker = TuckerTensor([d, d, d, d], r, 'symmetric', false, ...
                        'init_method', 'orthogonal');
T_tucker.G = randn(r, r, r, r) * 0.1;

% Up will be computed inside get_proj_grad_kronecker if needed

% Create operator
op = TuckerOperator(A_cells, 'order', 4, 'symmetric', false, 'dims', [d,d,d,d]);
op.A_mat = A_mat;

% Generate synthetic measurements
y_true = op.forward(T_tucker) + randn(m, 1) * 0.01;
y0 = op.forward(T_tucker);
residual = y0 - y_true;

fprintf('  Residual norm: %.6f\n\n', norm(residual));

%% Method 1: Current Implementation (Project-then-Sum)
fprintf('--- Method 1: Project-then-Sum (Current Implementation) ---\n');
fprintf('Computing gradient using get_proj_grad_kronecker...\n');

tic;
Grad_F_method1 = op.get_proj_grad_kronecker(T_tucker, y0, y_true);
time_method1 = toc;

fprintf('  Time: %.6f seconds\n', time_method1);
fprintf('  Core gradient norm: %.6e\n', norm(Grad_F_method1.G(:)));
fprintf('  Up{1} norm: %.6e\n', norm(Grad_F_method1.Up{1}(:)));
fprintf('  Up{2} norm: %.6e\n', norm(Grad_F_method1.Up{2}(:)));
fprintf('  Up{3} norm: %.6e\n', norm(Grad_F_method1.Up{3}(:)));
fprintf('  Up{4} norm: %.6e\n\n', norm(Grad_F_method1.Up{4}(:)));

%% Method 2: Reference (Sum-then-Project)
fprintf('--- Method 2: Sum-then-Project (Reference Full Tensor) ---\n');
fprintf('Building full Euclidean gradient tensor...\n');

% Extract components
G = T_tucker.G;
U1 = T_tucker.U{1};
U2 = T_tucker.U{2};
U3 = T_tucker.U{3};
U4 = T_tucker.U{4};

% Compute full Euclidean gradient: Σ_i residual_i * (A_i ⊗ A_i)
Grad_Euclidean = zeros(d, d, d, d);
for i = 1:m
    Ai = A_cells{i};
    % A_i ⊗ A_i as 4D tensor
    for i1 = 1:d
        for i2 = 1:d
            for i3 = 1:d
                for i4 = 1:d
                    Grad_Euclidean(i1, i2, i3, i4) = ...
                        Grad_Euclidean(i1, i2, i3, i4) + ...
                        residual(i) * Ai(i1, i2) * Ai(i3, i4);
                end
            end
        end
    end
end
fprintf('  Euclidean gradient norm: %.6e\n', norm(Grad_Euclidean(:)));

% Now project onto Tucker tangent space
fprintf('Projecting onto tangent space...\n');

%% Project Core Gradient
% Core gradient: G component after projecting with P_{U_k}
dG_method2 = zeros(r, r, r, r);
for j1 = 1:r
    for j2 = 1:r
        for j3 = 1:r
            for j4 = 1:r
                % Project: (U₁ ⊗ U₂ ⊗ U₃ ⊗ U₄)^T * Grad_Euclidean
                for i1 = 1:d
                    for i2 = 1:d
                        for i3 = 1:d
                            for i4 = 1:d
                                dG_method2(j1, j2, j3, j4) = ...
                                    dG_method2(j1, j2, j3, j4) + ...
                                    U1(i1, j1) * U2(i2, j2) * U3(i3, j3) * U4(i4, j4) * ...
                                    Grad_Euclidean(i1, i2, i3, i4);
                            end
                        end
                    end
                end
            end
        end
    end
end
fprintf('  Core gradient norm (method 2): %.6e\n', norm(dG_method2(:)));

%% Project Factor Gradients (Up components)
% Precompute G pseudoinverses
G_unfold = cell(4, 1);
G_pinv = cell(4, 1);
for k = 1:4
    perm = [k, setdiff(1:4, k)];
    G_perm = permute(G, perm);
    G_unfold{k} = reshape(G_perm, r, []);
    G_pinv{k} = pinv(G_unfold{k});
end

% Mode 1: Project onto Up{1}
fprintf('Computing Up gradients via full projection...\n');
grad_term_1_full = zeros(d, r^3);
% Contract Euclidean gradient with U2, U3, U4
for i1 = 1:d
    idx = 1;
    for j4 = 1:r
        for j3 = 1:r
            for j2 = 1:r
                for i2 = 1:d
                    for i3 = 1:d
                        for i4 = 1:d
                            grad_term_1_full(i1, idx) = grad_term_1_full(i1, idx) + ...
                                Grad_Euclidean(i1, i2, i3, i4) * ...
                                U2(i2, j2) * U3(i3, j3) * U4(i4, j4);
                        end
                    end
                end
                idx = idx + 1;
            end
        end
    end
end
P_perp_1_full = grad_term_1_full - U1 * (U1' * grad_term_1_full);
dUp1_method2 = P_perp_1_full * G_pinv{1};

% Mode 2: Project onto Up{2}
grad_term_2_full = zeros(d, r^3);
for i2 = 1:d
    idx = 1;
    for j4 = 1:r
        for j3 = 1:r
            for j1 = 1:r
                for i1 = 1:d
                    for i3 = 1:d
                        for i4 = 1:d
                            grad_term_2_full(i2, idx) = grad_term_2_full(i2, idx) + ...
                                Grad_Euclidean(i1, i2, i3, i4) * ...
                                U1(i1, j1) * U3(i3, j3) * U4(i4, j4);
                        end
                    end
                end
                idx = idx + 1;
            end
        end
    end
end
P_perp_2_full = grad_term_2_full - U2 * (U2' * grad_term_2_full);
dUp2_method2 = P_perp_2_full * G_pinv{2};

% Mode 3: Project onto Up{3}
grad_term_3_full = zeros(d, r^3);
for i3 = 1:d
    idx = 1;
    for j4 = 1:r
        for j2 = 1:r
            for j1 = 1:r
                for i1 = 1:d
                    for i2 = 1:d
                        for i4 = 1:d
                            grad_term_3_full(i3, idx) = grad_term_3_full(i3, idx) + ...
                                Grad_Euclidean(i1, i2, i3, i4) * ...
                                U1(i1, j1) * U2(i2, j2) * U4(i4, j4);
                        end
                    end
                end
                idx = idx + 1;
            end
        end
    end
end
P_perp_3_full = grad_term_3_full - U3 * (U3' * grad_term_3_full);
dUp3_method2 = P_perp_3_full * G_pinv{3};

% Mode 4: Project onto Up{4}
grad_term_4_full = zeros(d, r^3);
for i4 = 1:d
    idx = 1;
    for j3 = 1:r
        for j2 = 1:r
            for j1 = 1:r
                for i1 = 1:d
                    for i2 = 1:d
                        for i3 = 1:d
                            grad_term_4_full(i4, idx) = grad_term_4_full(i4, idx) + ...
                                Grad_Euclidean(i1, i2, i3, i4) * ...
                                U1(i1, j1) * U2(i2, j2) * U3(i3, j3);
                        end
                    end
                end
                idx = idx + 1;
            end
        end
    end
end
P_perp_4_full = grad_term_4_full - U4 * (U4' * grad_term_4_full);
dUp4_method2 = P_perp_4_full * G_pinv{4};

fprintf('  Up{1} norm (method 2): %.6e\n', norm(dUp1_method2(:)));
fprintf('  Up{2} norm (method 2): %.6e\n', norm(dUp2_method2(:)));
fprintf('  Up{3} norm (method 2): %.6e\n', norm(dUp3_method2(:)));
fprintf('  Up{4} norm (method 2): %.6e\n\n', norm(dUp4_method2(:)));

%% Compare Results
fprintf('=== Comparison: Method 1 vs Method 2 ===\n\n');

% Core gradient comparison
error_G = norm(Grad_F_method1.G(:) - dG_method2(:)) / norm(dG_method2(:));
fprintf('Core Gradient (G):\n');
fprintf('  ||G_method1 - G_method2|| / ||G_method2||: %.6e\n', error_G);
if error_G < 1e-10
    fprintf('  ✓ PASS: Core gradients match\n\n');
else
    fprintf('  ✗ FAIL: Core gradients do not match!\n\n');
end

% Factor gradient comparisons
error_Up1 = norm(Grad_F_method1.Up{1}(:) - dUp1_method2(:)) / norm(dUp1_method2(:));
error_Up2 = norm(Grad_F_method1.Up{2}(:) - dUp2_method2(:)) / norm(dUp2_method2(:));
error_Up3 = norm(Grad_F_method1.Up{3}(:) - dUp3_method2(:)) / norm(dUp3_method2(:));
error_Up4 = norm(Grad_F_method1.Up{4}(:) - dUp4_method2(:)) / norm(dUp4_method2(:));

fprintf('Factor Gradients (Up):\n');
fprintf('  Up{1}: ||method1 - method2|| / ||method2||: %.6e', error_Up1);
if error_Up1 < 1e-10
    fprintf(' ✓\n');
else
    fprintf(' ✗\n');
end

fprintf('  Up{2}: ||method1 - method2|| / ||method2||: %.6e', error_Up2);
if error_Up2 < 1e-10
    fprintf(' ✓\n');
else
    fprintf(' ✗\n');
end

fprintf('  Up{3}: ||method1 - method2|| / ||method2||: %.6e', error_Up3);
if error_Up3 < 1e-10
    fprintf(' ✓\n');
else
    fprintf(' ✗\n');
end

fprintf('  Up{4}: ||method1 - method2|| / ||method2||: %.6e', error_Up4);
if error_Up4 < 1e-10
    fprintf(' ✓\n');
else
    fprintf(' ✗\n');
end

%% Form Complete Gradient Tensors for Direct Comparison
fprintf('\n--- Forming Complete Result Tensors ---\n');
fprintf('Reconstructing full gradient tensors from Tucker components...\n');

% Method 1: Form full tensor from Method 1 results
% Gradient in Tucker form with components (G, Up)
% Full tensor: Σ_k G ×_k Up_k (sum over all modes)
fprintf('Method 1 - Forming tensor from get_proj_grad_kronecker result...\n');

% The tangent space representation is: dG at core + Σ_k (G ×_k dUp_k)
% Core contribution
Grad_full_method1 = zeros(d, d, d, d);
for i1 = 1:d
    for i2 = 1:d
        for i3 = 1:d
            for i4 = 1:d
                % Contribution from core gradient: G ×_1 U1 ×_2 U2 ×_3 U3 ×_4 U4
                % where G is the gradient core
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method1(i1, i2, i3, i4) = ...
                                    Grad_full_method1(i1, i2, i3, i4) + ...
                                    Grad_F_method1.G(j1, j2, j3, j4) * ...
                                    U1(i1, j1) * U2(i2, j2) * U3(i3, j3) * U4(i4, j4);
                            end
                        end
                    end
                end
                
                % Contribution from Up{1}: G ×_1 Up1 ×_2 U2 ×_3 U3 ×_4 U4
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method1(i1, i2, i3, i4) = ...
                                    Grad_full_method1(i1, i2, i3, i4) + ...
                                    G(j1, j2, j3, j4) * ...
                                    Grad_F_method1.Up{1}(i1, j1) * U2(i2, j2) * U3(i3, j3) * U4(i4, j4);
                            end
                        end
                    end
                end
                
                % Contribution from Up{2}: G ×_1 U1 ×_2 Up2 ×_3 U3 ×_4 U4
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method1(i1, i2, i3, i4) = ...
                                    Grad_full_method1(i1, i2, i3, i4) + ...
                                    G(j1, j2, j3, j4) * ...
                                    U1(i1, j1) * Grad_F_method1.Up{2}(i2, j2) * U3(i3, j3) * U4(i4, j4);
                            end
                        end
                    end
                end
                
                % Contribution from Up{3}: G ×_1 U1 ×_2 U2 ×_3 Up3 ×_4 U4
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method1(i1, i2, i3, i4) = ...
                                    Grad_full_method1(i1, i2, i3, i4) + ...
                                    G(j1, j2, j3, j4) * ...
                                    U1(i1, j1) * U2(i2, j2) * Grad_F_method1.Up{3}(i3, j3) * U4(i4, j4);
                            end
                        end
                    end
                end
                
                % Contribution from Up{4}: G ×_1 U1 ×_2 U2 ×_3 U3 ×_4 Up4
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method1(i1, i2, i3, i4) = ...
                                    Grad_full_method1(i1, i2, i3, i4) + ...
                                    G(j1, j2, j3, j4) * ...
                                    U1(i1, j1) * U2(i2, j2) * U3(i3, j3) * Grad_F_method1.Up{4}(i4, j4);
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf('  Full tensor norm (method 1): %.6e\n', norm(Grad_full_method1(:)));

% Method 2: Form full tensor from Method 2 results
fprintf('Method 2 - Forming tensor from full projection result...\n');

Grad_full_method2 = zeros(d, d, d, d);
for i1 = 1:d
    for i2 = 1:d
        for i3 = 1:d
            for i4 = 1:d
                % Contribution from core gradient
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method2(i1, i2, i3, i4) = ...
                                    Grad_full_method2(i1, i2, i3, i4) + ...
                                    dG_method2(j1, j2, j3, j4) * ...
                                    U1(i1, j1) * U2(i2, j2) * U3(i3, j3) * U4(i4, j4);
                            end
                        end
                    end
                end
                
                % Contribution from Up{1}
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method2(i1, i2, i3, i4) = ...
                                    Grad_full_method2(i1, i2, i3, i4) + ...
                                    G(j1, j2, j3, j4) * ...
                                    dUp1_method2(i1, j1) * U2(i2, j2) * U3(i3, j3) * U4(i4, j4);
                            end
                        end
                    end
                end
                
                % Contribution from Up{2}
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method2(i1, i2, i3, i4) = ...
                                    Grad_full_method2(i1, i2, i3, i4) + ...
                                    G(j1, j2, j3, j4) * ...
                                    U1(i1, j1) * dUp2_method2(i2, j2) * U3(i3, j3) * U4(i4, j4);
                            end
                        end
                    end
                end
                
                % Contribution from Up{3}
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method2(i1, i2, i3, i4) = ...
                                    Grad_full_method2(i1, i2, i3, i4) + ...
                                    G(j1, j2, j3, j4) * ...
                                    U1(i1, j1) * U2(i2, j2) * dUp3_method2(i3, j3) * U4(i4, j4);
                            end
                        end
                    end
                end
                
                % Contribution from Up{4}
                for j1 = 1:r
                    for j2 = 1:r
                        for j3 = 1:r
                            for j4 = 1:r
                                Grad_full_method2(i1, i2, i3, i4) = ...
                                    Grad_full_method2(i1, i2, i3, i4) + ...
                                    G(j1, j2, j3, j4) * ...
                                    U1(i1, j1) * U2(i2, j2) * U3(i3, j3) * dUp4_method2(i4, j4);
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf('  Full tensor norm (method 2): %.6e\n', norm(Grad_full_method2(:)));

% Compare full tensors
error_full = norm(Grad_full_method1(:) - Grad_full_method2(:)) / norm(Grad_full_method2(:));
fprintf('\nFull Tensor Comparison:\n');
fprintf('  ||Grad_full_method1 - Grad_full_method2|| / ||Grad_full_method2||: %.6e\n', error_full);

if error_full < 1e-10
    fprintf('  ✓ PASS: Full gradient tensors match\n\n');
else
    fprintf('  ✗ FAIL: Full gradient tensors do not match!\n\n');
end

%% Summary
fprintf('\n=== Test Summary ===\n');
max_error_components = max([error_G, error_Up1, error_Up2, error_Up3, error_Up4]);
fprintf('Maximum relative error across components: %.6e\n', max_error_components);
fprintf('Full tensor relative error: %.6e\n', error_full);

if error_full < 1e-10 && max_error_components < 1e-10
    fprintf('\n✓ ALL TESTS PASSED\n');
    fprintf('Project-then-sum equals sum-then-project (as expected by linearity)\n');
    fprintf('Both component-wise and full tensor comparisons match perfectly!\n');
else
    fprintf('\n✗ SOME TESTS FAILED\n');
    fprintf('There may be an error in the gradient computation!\n');
    if error_full < 1e-10 && max_error_components >= 1e-10
        fprintf('Note: Full tensors match but components differ - check Tucker representation.\n');
    elseif error_full >= 1e-10 && max_error_components < 1e-10
        fprintf('Note: Components match but full tensors differ - check reconstruction.\n');
    end
end

fprintf('\nConclusion:\n');
fprintf('The efficient Kronecker implementation correctly computes the\n');
fprintf('Riemannian gradient by projecting each measurement contribution\n');
fprintf('and summing, which is equivalent to computing the full Euclidean\n');
fprintf('gradient and then projecting onto the tangent space.\n');
fprintf('\nVerification includes:\n');
fprintf('  1. Component-wise comparison (G, Up{1-4})\n');
fprintf('  2. Full tensor reconstruction and comparison\n');
