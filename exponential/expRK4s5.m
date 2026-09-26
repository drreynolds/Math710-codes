function [t, expRK4s5_sol] = expRK4s5(F, A, g, t0, t_end, u0, N)
% Fourth-order, five-stage exponential Runge--Kutta method.
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
        [incr1, ~] = kiops(1, 0.5*dt*A, input_mat_1);
        Un2 = u(:) + 0.5*dt*incr1(:);
        Dn2 = g(Un2) - g(u);

        % Compute Dn3.
        input_mat_2 = [zero; 0.5*Fu(:).'; Dn2(:).'];
        [incr2, ~] = kiops(1, 0.5*dt*A, input_mat_2);
        Un3 = u(:) + dt*incr2(:);
        Dn3 = g(Un3) - g(u);

        % Compute Dn4.
        input_mat_3 = [zero; Fu(:).'; (Dn2(:).' + Dn3(:).')];
        [incr3, ~] = kiops(1, dt*A, input_mat_3);
        Un4 = u(:) + dt*incr3(:);
        Dn4 = g(Un4) - g(u);

        % Compute Dn5.
        input_mat_4 = [zero; 0.5*Fu(:).'; 0.25*(2*Dn2(:).' + 2*Dn3(:).' - Dn4(:).'); 0.5*(-Dn2(:).' - Dn3(:).' + Dn4(:).')];
        [incr4, ~] = kiops(1, 0.5*dt*A, input_mat_4);
        input_mat_5 = [zero; zero; 0.25*(Dn2(:).' + Dn3(:).' - Dn4(:).'); (-Dn2(:).' - Dn3(:).' + Dn4(:).')];
        [incr5, ~] = kiops(1, dt*A, input_mat_5);
        Un5 = u(:) + dt*incr4(:) + dt*incr5(:);
        Dn5 = g(Un5) - g(u);

        % Update u.
        input_mat_6 = [zero; Fu(:).'; (-Dn4(:).' + 4*Dn5(:).'); (4*Dn4(:).' - 8*Dn5(:).')];
        [incr6, ~] = kiops(1, dt*A, input_mat_6);
        u = u(:) + dt*incr6(:);
    end

    expRK4s5_sol = u;
end
