function [t_values, y] = RK2(f, t0, t_end, y0, NTS)
% Classical second-order Runge--Kutta reference method.
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
        k2 = f(y + h*k1);
        y = y + h*0.5*(k1 + k2);
        t = t + h;
        idx = idx + 1;
        t_values(idx) = t;
    end
    t_values = t_values(1:idx);
end
