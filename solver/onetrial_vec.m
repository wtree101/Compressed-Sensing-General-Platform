function [Error_Stand, Error_function] = onetrial_vec(params)
    % Vector Recovery Trial Function
    % Inputs:
    %   params - Parameter structure containing all parameters
    % Outputs:
    %   Error_Stand - Standard error tracking
    %   Error_function - Function error tracking
    
    % Required parameters
    if ~isfield(params, 'm'), error('Parameter m is required'); end
    if ~isfield(params, 'sparsity'), error('Parameter sparsity is required'); end
    if ~isfield(params, 'kappa'), error('Parameter kappa is required'); end
    if ~isfield(params, 'd1'), error('Parameter d1 is required'); end
    
    % Extract core parameters
    m = params.m;
    sparsity = params.sparsity;
    kappa = params.kappa;
    d1 = params.d1;
    % For vectors, d2 is not needed, but kept for compatibility
    if isfield(params, 'd2')
        d2 = params.d2;
    else
        d2 = 1; % Default for vector case
    end
    
    % Optional parameters with defaults
    if isfield(params, 'verbose')
        verbose = params.verbose;
    else
        verbose = 0;
    end
    
    if isfield(params, 'xstar')
        xstar = params.xstar;
    else
        % Generate sparse ground truth vector
        xstar = generate_sparse_vector(d1, sparsity, kappa);
    end
    
    if isfield(params, 'init_flag')
        init_flag = params.init_flag;
    else
        init_flag = 1;
    end
    
    if isfield(params, 'problem_flag')
        problem_flag = params.problem_flag;
    else
        problem_flag = 0; % default to sensing
    end

    %% Generate sensing operator
    A = generate_A_vec(problem_flag, m, d1, params);

    % Create operator structure for vectors
    operator.A = @(x) A * x ;  % Forward operator: vector to measurements
    operator.A_star = @(y_vec) A' * y_vec ;  % Adjoint operator: measurements to vector

    y = operator.A(xstar) / sqrt(m);
    % Apply nonlinear transformation if specified
    if isfield(params, 'nonlinear_func') && ~isempty(params.nonlinear_func)
        y = params.nonlinear_func(y);
    end

    %% Initialization
    % Use the initialization function handle for vectors
    if isfield(params, 'init') && ~isempty(params.init)
        % Use the function handle from set_init_vec
        if init_flag == 0
            xl = params.init(y, operator, d1, sparsity, m);
        else
            xl = params.init(y, operator, d1, sparsity, m, params.init_scale);
        end
    else
        % Default initialization
        xl = randn(d1, 1) * params.init_scale;
        disp('Using default vector initialization')
    end

    %% Solve using vector-compatible solver
    % Use the solver function handle
    if isfield(params, 'alg') && ~isempty(params.alg)
        % Use the modular solver for vectors
        [Error_Stand, Error_function] = params.alg(xl, [], y, operator, d1, 1, sparsity, m, params);
    else
        % Fallback to gradient descent if no solver specified
        [Error_Stand, Error_function] = solve_GD_vec(xl, [], y, operator, d1, 1, sparsity, m, params);
    end

    %% Test and visualization
    if (verbose == 1)
        semilogy(Error_Stand)
        title('Vector Recovery Error Progress')
        xlabel('Iteration')
        ylabel('Relative Error')
    end
end

function xstar = generate_sparse_vector(d1, sparsity, kappa)
    % Generate sparse ground truth vector
    % Inputs:
    %   d1 - Vector dimension
    %   sparsity - Number of non-zero elements
    %   kappa - Signal strength parameter
    
    xstar = zeros(d1, 1);
    
    % Randomly select sparsity locations
    support = randperm(d1, min(sparsity, d1));
    
    % Generate non-zero values with condition number kappa
    if sparsity > 0
        values = randn(sparsity, 1);
        values = values / norm(values) * sqrt(sparsity);
        
        % Apply condition number scaling
        if kappa > 1
            values = values .* (1 + (kappa-1) * rand(sparsity, 1));
        end
        
        xstar(support) = values;
    end
end

function A = generate_A_vec(problem_flag, m, d1, params)
    % Generate sensing matrix for vector problems
    % Inputs:
    %   problem_flag - Type of sensing problem
    %   m - Number of measurements
    %   d1 - Vector dimension
    %   params - Parameter structure
    
    switch problem_flag
        case 0 % Standard Gaussian sensing
            A = randn(m, d1) / sqrt(d1);
            
        case 1 % Phase retrieval (magnitude measurements)
            A = randn(m, d1) / sqrt(d1);
            % Note: nonlinear function should be set to abs() for phase retrieval
            
        case 2 % Symmetric Gaussian sensing
            A = randn(m, d1);
            A = (A + A') / 2; % Make symmetric
            A = A / sqrt(d1);
            
        case 3 % Custom sensing (if provided in params)
            if isfield(params, 'A_vec')
                A = params.A_vec;
            else
                error('Custom sensing matrix A_vec must be provided in params for problem_flag = 3');
            end
            
        case 4 % Fourier sensing (partial DFT)
            idx = randperm(d1, m);
            F = dftmtx(d1) / sqrt(d1);
            A = F(idx, :);
            
        otherwise
            error('Unknown problem_flag for vector sensing');
    end
end
