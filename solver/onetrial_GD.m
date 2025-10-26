function  [Error_Stand, Error_function] = onetrial_GD(m,r,kappa,lambda,params)
    %todo: add lambda
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

if isfield(params, 'problem_flag')
    problem_flag = params.problem_flag;
else
    problem_flag = 0; % default to sensing
end

if problem_flag == 0
    % random sensing matrix %sym
    A = normrnd(0,1,m,d1*d2);
elseif problem_flag == 1
    % phase retrieval: d1 == d2, each row of A is vec(aa^T), a is d1 Gaussian vector
    if d1 ~= d2
        error('For phase retrieval, d1 must equal d2.');
    end
    A = zeros(m, d1*d2);
    for i = 1:m
        a = normrnd(0,1,d1,1);
        A(i,:) = reshape(a*a', 1, []);
    end
elseif problem_flag == 2
    if d1 ~= d2
        error('For symmetric sensing, d1 must equal d2.');
    end
    % sensing with symmetric pointwise Gaussian
    A = normrnd(0,1,m,d1*d2);
    % symmetrize each row to correspond to symmetric matrices
    for i = 1:m
        Ai = reshape(A(i,:), d1, d2);
        Ai = (Ai + Ai')/2;
        A(i,:) = Ai(:)';
    end
elseif problem_flag == 3
    if d1 ~= d2
        error('For Richard''s example, d1 must equal d2.');
    end
    % Richard's example
    %[A, Xstar] = prob3(d1, 4, r, r);
    %Xstar = Xstar / norm(Xstar, 'fro'); % Normalize Xstar
    A = params.A;
elseif problem_flag == 4
    A = zeros(m, d1*d2);
    for i = 1:m
        a = normrnd(0,1,d1,d1)/sqrt(d1);
        A(i,:) = reshape(a*a', 1, []);
    end

else

    error('Unknown problem_flag value. Use 0 for sensing, 1 for phase retrieval.');
end
y = A*Xstar(:)/sqrt(m);
% 
if init_flag == 0
[X0,U0,S0,V0] = Initialization(y,A,d1,d2,r,m);
Xl = X0;
Ul = U0(:,1:r)*sqrt(S0(1:r,1:r));
    if (norm(Xl - Ul*Ul')>1)
        display('warn Init!!')
    end
else
    [X0,U0] = Initialization_random(y,A,d1,d2,r,m,params.init_scale);
    Ul = U0;
    Xl = X0;
end





% Error Tracking
Error_Stand = zeros(T,1);
Error_Stand(1) = norm(X0-Xstar,'fro');
Error_function = zeros(T,1);
Error_function(1) = norm(y - A*X0(:),'fro')/norm(y);

%disp(norm(X0,'fro'));
%s = zeros(m,1);

% standard RGD
for l = 2:T
    % compute Gl
    
    yl = A * Xl(:) / sqrt(m);
    
    s = y - (A * Xl(:)) / sqrt(m);
  
    Ul = Ul + mu* ( (1*(1/sqrt(m)) * reshape(A' * s, [d1, d2]))*Ul - lambda*Ul); 
    %Xl_new = Ul * Ul'; % Update Xl
    % Track Errors
    
    % Swap 
    Xl = Ul*Ul';
    Error_function(l) = norm(yl - y, 'fro')^2 + lambda * (norm(Xl, 'fro')^2 -  norm(Xstar, 'fro')^2) / 2;
    Error_Stand(l) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
end


%test
    if (verbose==1)
        semilogy(Error_Stand)
    end
    %Is_success = (norm(Xl - Xstar) < 1e-2);
    
    %err = norm(Xl - Xstar, 'fro') / norm(Xstar, 'fro');
    

end

