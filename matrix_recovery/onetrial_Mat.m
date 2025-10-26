function [output, is_success] = onetrial_Mat(params)
    % ONETRIAL Single trial of low-rank matrix recovery
    %
    % Inputs:
    %   params - Structure with required and optional fields
    %
    % Outputs:
    %   output      - Struct with auxiliary information:
    %                 .Error_Stand    - Relative error history vs ground truth
    %                 .Error_function - Loss function history
    %                 .Xl_final       - Final recovered matrix
    %                 .final_error    - Final relative error
    %                 .recovered_rank - Rank of recovered matrix
    %   is_success  - Binary flag: 1 if recovered successfully, 0 otherwise
    
    % Required parameters
    if ~isfield(params, 'm'), error('Parameter m is required'); end
    if ~isfield(params, 'r'), error('Parameter r is required'); end
    if ~isfield(params, 'kappa'), error('Parameter kappa is required'); end
    if ~isfield(params, 'd1'), error('Parameter d1 is required'); end
    if ~isfield(params, 'd2'), error('Parameter d2 is required'); end
    
    % Extract core parameters
    m = params.m;
    r = params.r;
    kappa = params.kappa;
    d1 = params.d1;
    d2 = params.d2;
    % Optional parameters with defaults
    if isfield(params, 'verbose')
        verbose = params.verbose;
    else
        verbose = 0;
    end
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
    else
        Xstar = groundtruth(d1, d2, r, kappa);
    end
    if isfield(params, 'init_flag')
        init_flag = params.init_flag;
    else
        init_flag = 1;
    end
    params.projection = @(X) rank_projection(X, r);

if isfield(params, 'problem_flag')
    problem_flag = params.problem_flag;
else
    problem_flag = 0; % default to sensing
end

%%
A = generate_A(problem_flag, m, d1, d2, params);

% Create operator structure
operator.A = @(X) A * X(:);  % Forward operator: matrix to measurements
operator.A_star = @(y_vec) reshape(A' * y_vec, [d1, d2]);  % Adjoint operator: measurements to matrix
% operator contains randomness specifc to this trial, 
% non-linear function is a global setting and set ealier

y = operator.A(Xstar) / sqrt(m);
% Apply nonlinear transformation if specified
if isfield(params, 'nonlinear_func') && ~isempty(params.nonlinear_func)
    y = params.nonlinear_func(y);
end

% Initialization with unified interface
% All initialization functions follow: [X0, U0, history] = func(y, operator, d1, d2, params)
if isfield(params, 'init') && ~isempty(params.init)
    % Prepare initialization parameters
    init_params = struct(params);
    init_params.T_power = get_param(params,'T_power',20);
    % Call unified initialization function
    [Xl, Ul, ~] = params.init(y, operator, d1, d2, init_params);
else
    % Default: random initialization
    disp('No Init!! Using random initialization');
    init_params = params;
    if ~isfield(init_params, 'scale')
        init_params.scale = 0.1; % Default scale
    end
    [Xl, Ul, ~] = Initialization_random(y, operator, d1, d2, init_params);
end

% Use the solver function handle
if isfield(params, 'alg_func') && ~isempty(params.alg_func)
    % Use the modular solver
    [solver_output, Xl] = params.alg_func(Xl, Ul, y, operator, d1, d2, r, m, params);
    Error_Stand = solver_output.Error_Stand;
    Error_function = solver_output.Error_function;
else
    % Fallback to RGD if no solver specified
    [solver_output, Xl] = solve_RGD(Xl, Ul, y, operator, d1, d2, r, m, params);
    Error_Stand = solver_output.Error_Stand;
    Error_function = solver_output.Error_function;
end

%% Final Analysis and Success Check
% Compute final error with sign rectification
[final_error, Xl_aligned] = rectify_sign_ambiguity(Xl, Xstar);

% Check success criterion
is_success = (final_error < 1e-2);

% Pack output struct
output = struct();
output.Error_Stand = Error_Stand;
output.Error_function = Error_function;
output.Xl_final = Xl_aligned;
output.final_error = final_error;
output.recovered_rank = rank(Xl, 1e-6);

% Optional: plot convergence
if verbose == 1
    semilogy(Error_Stand)
end

end

