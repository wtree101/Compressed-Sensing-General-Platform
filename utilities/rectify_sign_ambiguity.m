function [error_rectified, Xl_rectified] = rectify_sign_ambiguity(Xl, Xstar)
% rectify_sign_ambiguity: Compute the best distance considering sign ambiguity
%
% In phase retrieval, the solution has a natural sign ambiguity: 
% if X is a solution, then -X is also a solution (since |A(-X)| = |A(X)|).
% This function computes the minimum error over {+X, -X}.
%
% Inputs:
%   Xl     - Estimated matrix/vector
%   Xstar  - Ground truth matrix/vector
%
% Outputs:
%   error_rectified - Minimum relative error: min(||Xl - Xstar||, ||Xl + Xstar||) / ||Xstar||
%   Xl_rectified    - The best aligned version of Xl (either Xl or -Xl)

    % Compute errors for both signs
    error_positive = norm(Xl - Xstar, 'fro');
    error_negative = norm(Xl + Xstar, 'fro');
    
    % Choose the sign that gives minimum error
    if error_positive <= error_negative
        error_rectified = error_positive / norm(Xstar, 'fro');
        Xl_rectified = Xl;
    else
        error_rectified = error_negative / norm(Xstar, 'fro');
        Xl_rectified = -Xl;
    end
end
