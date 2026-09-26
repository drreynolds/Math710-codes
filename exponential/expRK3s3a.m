function [t, expRK3s3_sol] = expRK3s3a(F, A, g, t0, t_end, u0, N)
% Third-order, three-stage exponential Runge--Kutta method.
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

        % Compute Dn2.
        input_mat_1 = [zero; Fu(:).'];
        [incr1, ~] = kiops(dt*(1/3), A, input_mat_1);
        Un2 = u(:) + incr1(:);
        Dn2 = g(Un2) - g(u);

        % Compute Dn3.
        input_mat_2 = [zero; Fu(:).'; (Dn2(:).'/dt*3)];
        [incr2, ~] = kiops(dt*(2/3), A, input_mat_2);
        Un3 = u(:) + incr2(:);
        Dn3 = g(Un3) - g(u);

        % Update u.
        input_mat_3 = [zero; Fu(:).'; (Dn3(:).'/dt*3/2)];
        [incr3, ~] = kiops(dt, A, input_mat_3);
        u = u(:) + incr3(:);
    end

    expRK3s3_sol = u;
end
