function [X0, U0, history] = Initialization(y, operator, d1, d2, params)
% Initialization: Spectral initialization using truncated SVD
%
% This function computes an initial estimate by forming the matrix
% X0 = A' * y and then applying truncated SVD to obtain a rank-r approximation.
%
% Inputs:
%   y         - The vector of magnitude measurements.
%   operator  - A struct containing the forward (A) and adjoint (A_star) operators.
%   d1, d2    - Dimensions of the signal (d1 x d2 matrix).
%   params    - A struct containing:
%               - r: Target rank for truncation
%               - m: Number of measurements
%               - Xstar: (optional) ground truth for error tracking
%
% Outputs:
%   X0        - The initialized matrix (d1 x d2).
%   U0        - Left singular vectors (d1 x r) for factorization compatibility
%   history   - Struct containing convergence history (minimal for this method)

    fprintf('--- Running Spectral Initialization (SVD) ---\n');
    
    % Extract parameters
    if ~isfield(params, 'r')
        error('Parameter r (rank) is required');
    end
    if ~isfield(params, 'm')
        error('Parameter m (measurements) is required');
    end
    
    r = params.r;
    m = params.m;
    has_ground_truth = isfield(params, 'Xstar') && ~isempty(params.Xstar);
    
    % Initialize history tracking
    history = struct();
    history.method = 'SVD';
    
    % Form initial matrix: X0 = A' * y / sqrt(m)
    X0 = operator.A_star(y) / sqrt(m);
    
    % Apply truncated SVD
    [U_full, S_full, V_full] = svd(X0);
    U0 = U_full(:, 1:r);
    S0 = S_full(1:r, 1:r);
    V0 = V_full(:, 1:r);
    
    % Rank-r approximation
    X0 = U0 * S0 * V0';
    
    % Store SVD components in history
    history.U0 = U0;
    history.S0 = S0;
    history.V0 = V0;
    history.singular_values = diag(S_full);
    
    % Track error if ground truth is available
    if has_ground_truth
        Xstar = params.Xstar;
        [history.error, ~] = rectify_sign_ambiguity(X0, Xstar);
        fprintf('Initialization error: %.4e\n', history.error);
    end
    
    fprintf('--- Spectral Initialization Complete ---\n');
    fprintf('Rank-%d approximation computed\n', r);
    fprintf('Top singular values: [%.4e, %.4e, ...]\n', history.singular_values(1), history.singular_values(min(2,end)));
end

