function [t_values, y] = RK4(f, t0, t_end, y0, NTS)
% Classical fourth-order Runge--Kutta reference method.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

    t_values = zeros(NTS + 1, 1);
    t_values(1) = t0;
    t = t0;
    y = y0(:);
    h = (t_end - t0) / NTS;
    idx = 1;
    while t < t_end
        if t + h > t_end
            h = t_end - t;  % Adjust final step.
        end
        k1 = f(y);
        k2 = f(y + 0.5*h*k1);
        k3 = f(y + 0.5*h*k2);
        k4 = f(y + h*k3);
        y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4);
        t = t + h;
        idx = idx + 1;
        t_values(idx) = t;
    end
    t_values = t_values(1:idx);
end
