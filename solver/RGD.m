function [X_new,U_new,S_new,V_new] = RGD(U,S,V,Z,r)
%RGD Summary of this function goes here
%   input
%  Xl = USV'
%  Zl 
% Output
% Hr(Xl+Ptl*Zl)

GradU = Z'*U;
Y1 = GradU-V*(V'*GradU);
GradV = Z*V;
Y2 = GradV- U*(U'*GradV);

[Q1,R1] = qr(Y1,0);
[Q2,R2] = qr(Y2,0);


W = zeros(2*r,2*r);
W(1:r,1:r) = S + U'*Z*V;
W(1:r,r+1:2*r) = R1';
W(r+1:2*r,1:r) = R2;
%W(r+1:2*r,r+1:2*r) = eta*Q2'*(Pr*Lr*Qr')*Q1;  % add sth in normal space

[P,Lam,Q] = svd(W);
U_new = [U,Q2]*P(:,1:r);
S_new = Lam(1:r,1:r);
V_new = (Q(:,1:r)'*[V';Q1'])';
X_new = U_new*S_new*(V_new)';
end

