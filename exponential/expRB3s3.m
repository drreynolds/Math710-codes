function [t, expRB3s3_sol] = expRB3s3(F, A, J, g, t0, t_end, u0, N, X, c2) %#ok<INUSD>
% Third-order exponential Rosenbrock method.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

    dt = (t_end - t0) / N;
    t = linspace(t0, t_end, N + 1).';
    n = numel(u0);
    zero = zeros(1, n);
    u = u0(:);  % To avoid modifying the input.

    for i = 1:N
        % Compute Un2.
        Fu = F(u);
        Jn = A + J(u, X);
        input_mat_1 = [zero; Fu(:).'];
        [incr1, ~] = kiops(c2*dt, Jn, input_mat_1);
        Un2 = u(:) + incr1(:);

        % gn(Un2)-gn(u).
        Dn2 = F(Un2) - Jn * Un2 - F(u) + Jn * u;

        % Update u.
        input_mat_2 = [zero; Fu(:).'; zero; (Dn2(:).*(2/c2^2)/dt^2).'];
        [incr2, ~] = kiops(dt, Jn, input_mat_2);
        u = u(:) + incr2(:);
    end

    expRB3s3_sol = u;
end
