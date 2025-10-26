function [X0, U0, history] = Initialization_random(y, operator, d1, d2, params)
% Initialization_random: Random initialization for matrix recovery
%
% This function generates a random low-rank matrix initialization
% by creating a random factor U and setting X0 = U * U'.
%
% Inputs:
%   y         - The vector of magnitude measurements (not used, for interface consistency).
%   operator  - A struct containing operators (not used, for interface consistency).
%   d1, d2    - Dimensions of the signal (d1 x d2 matrix).
%   params    - A struct containing:
%               - r: Target rank for initialization
%               - scale: (optional) scaling factor for random initialization (default: 0.1)
%               - Xstar: (optional) ground truth for error tracking
%
% Outputs:
%   X0        - The initialized matrix (d1 x d2).
%   U0        - Random factor (d1 x r) such that X0 = U0 * U0'
%   history   - Struct containing initialization info

    fprintf('--- Running Random Initialization ---\n');
    
    % Extract parameters
    if ~isfield(params, 'r')
        error('Parameter r (rank) is required');
    end
    
    r = params.r;
    
    % Get scale parameter
    if isfield(params, 'scale')
        scale = params.scale;
    else
        scale = 0.1; % Default scale
    end
    
    has_ground_truth = isfield(params, 'Xstar') && ~isempty(params.Xstar);
    
    % Initialize history tracking
    history = struct();
    history.method = 'Random';
    history.scale = scale;
    
    % Generate random factor U
    U0 = randn(d1, r);
    U0 = U0 / norm(U0, 'fro'); % Normalize to have unit Frobenius norm
    U0 = U0 * scale;
    
    % Create symmetric rank-r initialization
    X0 = U0 * U0';
    
    % Store factor in history
    history.U0 = U0;
    history.rank = r;
    
    % Track error if ground truth is available
    if has_ground_truth
        Xstar = params.Xstar;
        [history.error, ~] = rectify_sign_ambiguity(X0, Xstar);
        fprintf('Initialization error: %.4e\n', history.error);
    end
    
    fprintf('--- Random Initialization Complete ---\n');
    fprintf('Generated rank-%d random matrix with scale %.4f\n', r, scale);
    fprintf('Frobenius norm: %.4e\n', norm(X0, 'fro'));
end

