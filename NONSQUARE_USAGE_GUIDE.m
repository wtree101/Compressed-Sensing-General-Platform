% QUICK GUIDE: Using Non-Square Matrices in Phase Diagram
%
% Now that initialize_tensor_lift_tucker_spectral supports non-square matrices,
% here's how to use it in Phasediagram_tensor_nonsym.m

%% 1. REMOVE the dimension equality constraint
% BEFORE (line ~23 in Phasediagram_tensor_nonsym.m):
% d2 = d1;  % Force square matrices

% AFTER:
d1 = 30;  % Row dimension
d2 = 20;  % Column dimension (can be different from d1)

%% 2. UPDATE ground truth generation
% BEFORE:
% U_true = randn(d1, r);
% Xstar = U_true * U_true';  % d1 × d1 symmetric matrix

% AFTER (for non-square):
U_true = randn(d1, r);
V_true = randn(d2, r);
Xstar = U_true * V_true';  % d1 × d2 non-symmetric matrix
Xstar = Xstar / norm(Xstar, 'fro');

% NOTE: For square matrices (d1 = d2), you can still use:
% U_true = randn(d1, r);
% Xstar = U_true * U_true';  % Creates symmetric matrix

%% 3. UPDATE measurement operator (if needed)
% The operator should already handle non-square matrices if properly defined:
operator.A = @(X) measurement_function(X);  % X is d1 × d2
operator.A_star = @(y) adjoint_function(y);  % Output is d1 × d2

% Example for Gaussian measurements:
A_matrix = randn(m, d1 * d2);
operator.A = @(X) A_matrix * X(:);
operator.A_star = @(y) reshape(A_matrix' * y, [d1, d2]);

%% 4. CALL initialization function
% No changes needed! It automatically handles non-square:
params = struct();
params.T_power = 20;
params.mu = 0.01;
params.r = 3;
params.Xstar = Xstar;
params.verbose = false;

[X0, ~, history] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params);

% X0 will be d1 × d2
% If d1 = d2, X0 will be symmetric
% If d1 ≠ d2, X0 will be non-symmetric

%% 5. EXAMPLE: Complete non-square setup
function example_nonsquare_phase_diagram()
    % Dimensions
    d1 = 30;  % Rows
    d2 = 20;  % Columns
    r = 2;    % Rank
    m = 5 * d1 * d2;  % Measurements
    
    % Ground truth (non-symmetric)
    U_true = randn(d1, r);
    V_true = randn(d2, r);
    Xstar = U_true * V_true';
    Xstar = Xstar / norm(Xstar, 'fro');
    
    % Measurements
    A_matrix = randn(m, d1 * d2);
    y = abs(A_matrix * Xstar(:)) / sqrt(m);  % Amplitude measurements
    
    % Operator
    operator = struct();
    operator.A = @(X) A_matrix * X(:);
    operator.A_star = @(y) reshape(A_matrix' * y, [d1, d2]);
    
    % Initialize
    params = struct();
    params.T_power = 20;
    params.mu = 0.01;
    params.r = 3;
    params.Xstar = Xstar;
    params.verbose = true;
    
    [X0, ~, history] = initialize_tensor_lift_tucker_spectral(y, operator, d1, d2, params);
    
    fprintf('Output size: %d × %d\n', size(X0, 1), size(X0, 2));
    fprintf('Final error: %.2e\n', history.final_error);
end

%% 6. GRID SETUP for phase diagrams
% Update grid generation to handle non-square:

% Example 1: Fix d2, vary d1
d2_fixed = 20;
d1_grid = [20, 30, 40, 50];  % Different row dimensions

% Example 2: Fix ratio, vary scale
ratio = 1.5;  % d1/d2 ratio
scale_grid = [10, 20, 30, 40];
d1_grid = round(scale_grid * ratio);
d2_grid = scale_grid;

% Example 3: Vary both independently
d1_grid = [20, 30, 40];
d2_grid = [15, 20, 25];
[D1, D2] = meshgrid(d1_grid, d2_grid);

%% KEY POINTS
% ✓ Square matrices (d1 = d2): Automatically symmetrized, backward compatible
% ✓ Non-square matrices (d1 ≠ d2): Not symmetrized, new capability
% ✓ Tucker rank adapts: min(5, floor(min(d1,d2)/4))
% ✓ Tensor dimensions: d1 × d2 × d1 × d2
% ✓ No code changes needed in calling code - just pass different d1, d2

%% COMPARISON: Square vs Non-Square

% Square Case (d1 = d2 = 20):
d1 = 20; d2 = 20;
U = randn(20, 2);
X_square = U * U';  % Symmetric
% Tensor: 20 × 20 × 20 × 20 = 160,000 elements
% Output: 20 × 20 symmetric matrix

% Non-Square Case (d1 = 30, d2 = 20):
d1 = 30; d2 = 20;
U = randn(30, 2);
V = randn(20, 2);
X_nonsquare = U * V';  % Non-symmetric
% Tensor: 30 × 20 × 30 × 20 = 360,000 elements
% Output: 30 × 20 non-symmetric matrix
