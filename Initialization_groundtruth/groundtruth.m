function X = groundtruth(d1,d2,r,kappa,symflag)


% %output PR transition graph
% U_seed = sign(rand(d1, r) - 0.5);
% [U, ~, ~] = svds(U_seed, r);
% V_seed = sign(rand(d2, r) - 0.5);
% [V, ~, ~] = svds(V_seed, r);
% 
% % S =diag(randn(r,1));
 diagonal_values = logspace(0, log10(kappa), r);
 %diagonal_values(r) = kappa;

    % Create the diagonal matrix
[U,~] = qr(randn(d1,r),0);
[V,~] = qr(randn(d2,r),0);
 S = diag(diagonal_values);
 if symflag == 0
    X = U*S*V';
 else
    X = U*S*U';
end

%X = randn(d1,r)*randn(r,d2);


X = X/norm(X,'fro');

%disp(cond_ratio(X,r));

end

