function Error_Stand = onetrial_RGD(m,r,kappa,lambda,params)

% random sensing matrix
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
else

    error('Unknown problem_flag value. Use 0 for sensing, 1 for phase retrieval.');
end


y = A*Xstar(:)/sqrt(m);
% 

if init_flag == 0
    [X0,U0,S0,V0] = Initialization(y,A,d1,d2,r,m);
    Xl = X0;
    Ul = U0(:,1:r);
    Sl = S0(1:r,1:r);
    Vl = V0(:,1:r);
else
    [X0,U0] = Initialization_random(y,A,d1,d2,r,m);
    [U0,S0,V0] = svd(X0);
    Xl = X0;
    Ul = U0(:,1:r);
    Sl = S0(1:r,1:r);
    Vl = V0(:,1:r);
end

%T = 10; %100 iter should be enough (and so quicker)

% Xl = X0;
% Ul = U0(:,1:r);
% Sl = S0(1:r,1:r);
% Vl = V0(:,1:r);
% Error Tracking
Error_Stand = zeros(T,1);
Error_Stand(1) = norm(X0-Xstar,'fro');
%disp(norm(X0,'fro'));

% standard RGD
for l = 2:T
    % compute Gl

    % for i = 1:m
    %     s(i)= y(i)-A(i,:)*Xl(:)/sqrt(m);
    % end
    s = y - (A * Xl(:)) / sqrt(m);
    Gl = zeros(d1,d2);
    % for i = 1:m
    %     Gl = Gl+(1/sqrt(m))*s(i)*reshape(A(i,:),[d1,d2]);
    % end
    Gl = Gl + 1*(1/sqrt(m)) * reshape(A' * s, [d1, d2]);
    
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


%test
    if (verbose==1)
        semilogy(Error_Stand)
    end
    %err = norm(Xl - Xstar, 'fro') / norm(Xstar, 'fro');


end

