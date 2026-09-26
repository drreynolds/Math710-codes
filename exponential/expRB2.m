function [t, expRB2_sol] = expRB2(F, A, J, g, t0, t_end, u0, N, X) %#ok<INUSD>
% Second-order exponential Rosenbrock method.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

    dt = (t_end - t0) / N;
    t = linspace(t0, t_end, N + 1).';
    n = numel(u0);
    zero = zeros(1, n);
    u = u0(:);  % To avoid modifying the input.

    for i = 1:N
        % Update u.
        Fu = F(u);
        Jn = A + J(u, X);
        input_mat_1 = [zero; Fu(:).'];
        [incr1, ~] = kiops(dt, Jn, input_mat_1);
        u = u(:) + incr1(:);
    end

    expRB2_sol = u;
end
