function A = generate_A_vec(problem_flag, m, d1, params)
    % Generate sensing matrix for vector problems
    % Inputs:
    %   problem_flag - Type of sensing problem
    %   m - Number of measurements
    %   d1 - Vector dimension
    %   params - Parameter structure
    
    switch problem_flag
        case 0 % Standard Gaussian sensing
            A = randn(m, d1) ;
            
        case 1 % Phase retrieval (magnitude measurements)
            A = randn(m, d1) ;
            % Note: nonlinear function should be set to abs() for phase retrieval
            
        case 2 % Symmetric Gaussian sensing
            A = randn(m, d1);
            A = (A + A') / 2; % Make symmetric
            A = A ;
            
        case 3 % Custom sensing (if provided in params)
            if isfield(params, 'A_vec')
                A = params.A_vec;
            else
                error('Custom sensing matrix A_vec must be provided in params for problem_flag = 3');
            end
            
        case 4 % Fourier sensing (partial DFT)
            idx = randperm(d1, m);
            F = dftmtx(d1) ;
            A = F(idx, :);
            
        otherwise
            error('Unknown problem_flag for vector sensing');
    end
end
