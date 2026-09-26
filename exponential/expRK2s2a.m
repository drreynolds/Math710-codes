function [t, expRK2_sol] = expRK2s2a(F, A, g, t0, t_end, u0, N)
% Second-order, two-stage exponential Runge--Kutta method.
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
        input_mat_1 = [zero; Fu(:).'];
        [incr1, ~] = kiops(dt, A, input_mat_1);
        Un2 = u(:) + incr1(:);

        % Compute Dn2.
        Dn2 = g(Un2) - g(u);
        input_mat_2 = [zero; zero; (Dn2(:).'/dt)];
        [incr2, ~] = kiops(dt, A, input_mat_2);

        % Update u.
        u = Un2 + incr2(:);
    end

    expRK2_sol = u;
end
