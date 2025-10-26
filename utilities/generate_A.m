function A = generate_A(problem_flag, m, d1, d2, params)
    switch problem_flag
        case 0
            % random sensing matrix %sym
            A = normrnd(0,1,m,d1*d2);
        case 1
            % phase retrieval: d1 == d2, each row of A is vec(aa^T), a is d1 Gaussian vector
            if d1 ~= d2
                error('For phase retrieval, d1 must equal d2.');
            end
            A = zeros(m, d1*d2);
            for i = 1:m
                a = normrnd(0,1,d1,1);
                A(i,:) = reshape(a*a', 1, []);
            end
        case 2
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
        case 3
            if d1 ~= d2
                error('For Richard''s example, d1 must equal d2.');
            end
            % Richard's example
            % [A, Xstar] = prob3(d1, 4, r, r);
            % Xstar = Xstar / norm(Xstar, 'fro'); % Normalize Xstar
            A = params.A;
        otherwise
            error('Unknown problem_flag value. Use 0 for random sensing, 1 for phase retrieval, 2 for symmetric sensing, 3 for Richard''s example.');
    end
end