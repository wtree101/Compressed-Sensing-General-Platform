function [X0, U0, history] = initialize_power_method(y, operator, d1, d2, params)
% INITIALIZE_POWER_METHOD Power method spectral initialization
%
% Performs spectral initialization using power iterations to find the
% principal eigenvector of Y = (1/m) * sum(y_i^2 * a_i * a_i').
% Optionally applies projection and scaling.
%
% Inputs:
%   y        - Measurement vector (m x 1)
%   operator - Struct with fields:
%              .A: Forward operator @(X) A*X(:)
%              .A_star: Adjoint operator @(y) reshape(A'*y, [d1,d2])
%   d1       - Matrix row dimension
%   d2       - Matrix column dimension
%   ~        - Unused (for interface consistency)
%   ~        - Unused (for interface consistency)
%   params   - Struct with optional fields:
%              .Xstar: Ground truth for error tracking
%              .projection: Projection function handle
%              .prefunc: Preprocessing function for measurements (default: @(y) y.^2)
%              .T_power: Number of power iterations (default: 20)
%
% Outputs:
%   X0       - Initialized matrix (d1 x d2)
%   U0       - Empty (for interface compatibility)
%   history  - Struct with convergence information:
%              .errors: Aligned error at each iteration (if Xstar provided)
%              .norms: Vector norms at each iteration
%              .iterations: Number of power iterations performed

% Extract parameters
if isfield(params, 'T_power')
    T_power = params.T_power;
else
    T_power = 40; % Default number of iterations
end

has_ground_truth = isfield(params, 'Xstar') && ~isempty(params.Xstar);
has_projection = isfield(params, 'projection') && ~isempty(params.projection);

% Initialize history tracking
history = struct();
if has_ground_truth
    history.errors = zeros(T_power, 1);
    Xstar = params.Xstar;
end
history.norms = zeros(T_power, 1);
history.iterations = T_power;

% Apply preprocessing function to measurements
if isfield(params, 'prefunc') && ~isempty(params.prefunc)
    y_processed = params.prefunc(y);
else
    y_processed = y.^2; % Default: square the measurements
end

v = get_param(params, 'Init',  ones(d1*d2, 1));
% Start with an all-ones initial guess
% v = ones(d1*d2, 1);
v = v / norm(v);

% Power iteration loop
for t = 1:T_power
    % Apply the operator: v_new = A' * processed(y) .* A * v
    Av = operator.A(reshape(v, [d1, d2]));
    w = y_processed .* Av;
    v_new = operator.A_star(w);
    
    % Apply projection if provided
    if has_projection
        v_new = params.projection(v_new);
    end
    
    v_new = v_new(:);
    
    % Record norm before normalization
    history.norms(t) = norm(v_new);
    
    % Normalize the vector for the next iteration
    v = v_new / norm(v_new);
    
    % Compute aligned error if ground truth is available
    if has_ground_truth
        X_current = reshape(v, [d1, d2]);
        [history.errors(t), ~] = rectify_sign_ambiguity(X_current, Xstar);
    end
end

% The principal eigenvector 'v' is the initialization
X0 = reshape(v, [d1, d2]);
U0 = [];  % Empty for compatibility

end
