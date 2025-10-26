function [output, Xl] = solve_RGD(Xl, Ul, y, operator, d1, d2, r, m, params)
    % RGD Solver - Riemannian Gradient Descent
    % Inputs:
    %   Xl, Ul - Initial matrices (or general objects)
    %   y - Measurement vector
    %   operator - Struct containing A (forward) and A_star (adjoint) operators
    %   d1, d2 - Matrix dimensions
    %   r - Rank
    %   m - Number of measurements
    %   params - Parameter structure containing mu, lambda, T, etc.
    % Outputs:
    %   output - Struct containing auxiliary information:
    %            .Error_Stand    - Standard error tracking
    %            .Error_function - Function error tracking
    %   Xl - Final recovered matrix
    
    % Extract parameters from params structure
    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
    if isfield(params, 'mu')
        mu = params.mu;
    else
        mu=0.2;
    end
    if isfield(params, 'lambda')
        lambda = params.lambda;
    else
        lambda=0;
    end
     if isfield(params, 'Xstar')
        Xstar = params.Xstar;
    else
        disp('Error: Xstar (ground truth) must be provided.');
    end
    
    %m = length(y);
    
    % Error Tracking
    Error_Stand = zeros(T,1);
    Error_Stand(1) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    Error_function = zeros(T,1);
    Error_function(1) = norm(y - operator.A(Xl)/sqrt(length(y)),'fro')/norm(y);
    
    % RGD optimization loop
    % standard RGD, need Ul, Sl and Vl
    [U0,S0,V0] = svd(Xl);
    Ul = U0(:,1:r);
    Sl = S0(1:r,1:r);
    Vl = V0(:,1:r);
    for l = 2:T
        % compute Gl - using operator structure
        s = y - operator.A(Xl) / sqrt(m);
        
        % Apply adjoint operator to get gradient
        Gl = mu * (1/sqrt(m)) * operator.A_star(s);
        
        % Ensure Gl is properly shaped as matrix
        if isvector(Gl)
            Gl = reshape(Gl, [d1, d2]);
        end
        
        % RGD
        [Xl_new,Ul_new,Sl_new,Vl_new] = RGD(Ul,Sl,Vl,Gl,r);
        
        % Track Errors
        Error_Stand(l) = norm(Xl_new-Xstar,'fro');
        if Error_Stand(l) > 1e3 %break
            Is_success = 0;
            return;
        end
        % Swap 
        Xl = Xl_new;
        Ul = Ul_new;
        Sl = Sl_new;
        Vl = Vl_new;
    end
    
    % Pack output struct
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
end
