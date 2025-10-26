function [output, Xl] = solve_AP(Xl, ~, y, operator, d1, d2, ~, ~, params)
    % AP Solver - Alternating Projection for Phase Retrieval
    % Inputs:
    %   Xl - Initial matrix (or general object)
    %   ~ - Unused (for compatibility)
    %   y - Measurement vector (magnitudes)
    %   operator - Struct containing A (forward) and A_star (adjoint) operators
    %   d1, d2 - Matrix dimensions
    %   ~ - Rank (unused, read from params)
    %   ~ - Number of measurements (unused, computed from y)
    %   params - Parameter structure containing T, nonlinear_func, etc.
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
    if isfield(params, 'nonlinear_func')
        nonlinear_func = params.nonlinear_func;
    else
        nonlinear_func = @(z) abs(z); % Default for phase retrieval
    end
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
    else
        disp('Error: Xstar (ground truth) must be provided.');
    end
    
    m = length(y);
    
    % Error Tracking
    Error_Stand = zeros(T,1);
    [Error_Stand(1), ~] = rectify_sign_ambiguity(Xl, Xstar);
    Error_function = zeros(T,1);
    
    % Initial measurement
    z = operator.A(Xl) / sqrt(m);
    Error_function(1) = norm(nonlinear_func(z) - y, 'fro')/norm(y, 'fro');
    
    % Alternating Projection optimization loop
    for l = 2:T
        % Step 1: Apply forward operator
        z = operator.A(Xl) / sqrt(m);
        
        % Step 2: Magnitude projection - preserve phases, replace magnitudes
        % For phase retrieval: z_new = y .* exp(1i * angle(z))
        % Since we're working with abs(z), we need to handle the phase
        phases = sign(z);
        z_projected = y .* phases;
        
        % Step 3: Find Xl_update such that A(Xl_update) ≈ z_projected
        % This is a least squares problem: minimize ||A(Xl) - z_projected||^2
        % Using pseudoinverse approach: Xl_update = pinv(A) * z_projected
        % For operator form: construct the Gram matrix A_star * A and solve
        % More robust approach using normal equations: (A^H A) Xl = A^H z_projected
        
        % Apply A^H to z_projected (scaled appropriately)
        % Create a hard copy of params to avoid modifying the original
        params_subproblem = struct(params);
        params_subproblem.T = 10; %use fewer iterations for subproblem
        params_subproblem.Xstar = Xl; % Use current estimate as reference for subproblem
        [~, Xl_update] = solve_PGD(Xl, [], z_projected, operator, d1, d2, 0, m, params_subproblem);
        
        % Step 4: Signal space projection (for low-rank matrix recovery)
        % Apply SVD for low-rank constraint if needed
        if isvector(Xl_update)
            Xl_update = reshape(Xl_update, [d1, d2]);
        end
        
        % Use params.projection if provided, else default SVD projection
        if isfield(params, 'projection')
            Xl = params.projection(Xl_update);
        else
           ...
        end
        
        % Track Errors
        z_new = operator.A(Xl) / sqrt(m);
        Error_function(l) = norm(nonlinear_func(z_new) - y, 'fro')/norm(y, 'fro');
        [Error_Stand(l), ~] = rectify_sign_ambiguity(Xl, Xstar);
    end
    
    % Pack output struct
    output = struct();
    output.Error_Stand = Error_Stand;
    output.Error_function = Error_function;
end
