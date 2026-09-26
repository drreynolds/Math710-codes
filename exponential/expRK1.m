function [t, expRK1_sol] = expRK1(F, A, g, t0, t_end, u0, N) %#ok<INUSD>
% First-order exponential Runge--Kutta method.
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
        u = u(:) + incr1(:);
    end

    expRK1_sol = u;
end
