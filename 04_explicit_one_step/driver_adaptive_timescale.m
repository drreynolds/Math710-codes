% Script that runs various adaptive methods on the nonlinear Kvaerno
% Prothero and Robinson problem:
%    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [      r'(t)/(2u)        ]
%    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2*sqrt(2+s(t))) ]
% where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5.
% This problem has analytical solution given by
%    u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).
%
% We use the parameters:
%   e = inter-variable coupling strength (0.5)
%   G = stiffness at slow time scale (-10)
%   w = variable time-scale separation factor (varies)
%
% This script uses adaptive explicit solvers to track the dynamical
% time scale as w is varied.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

Tf = 5;
Nt = 500;
tvals = linspace(0, Tf, Nt+1).';
epsilon = 0.5;
% Vary w to change the time-scale separation while keeping stiffness fixed.
W = [1, 10, 100, 1000, 10000];
G = -10;

% problem-defining functions
r = @(t) 0.5*cos(t);
s = @(t, w) cos(w*t);
rdot = @(t) -0.5*sin(t);
sdot = @(t, w) -w*sin(w*t);
utrue = @(t) sqrt(1 + r(t));
vtrue = @(t, w) sqrt(2 + s(t, w));
ytrue = @(t, w) [utrue(t), vtrue(t, w)];
f = @(t, y, w) [G, epsilon; epsilon, -1] * [(-1 + y(1)^2 - r(t)) / (2*y(1)); ...
                                            (-2 + y(2)^2 - s(t, w)) / (2*y(2))] ...
                + [rdot(t)/(2*y(1)); sdot(t, w)/(2*sqrt(2+s(t, w)))];

% create plots for visual diagnostics
figure(1);

for k = 1:numel(W)
    w = W(k);
    fprintf('\nKPR problem with w = %d\n\n', w);

    Y0 = ytrue(0, w);

    % compute and store the analytical solution
    Ytrue = ytrue(tvals, w);

    rtol = 1e-4;
    atol = 1e-11;
    % ERK32 adapts its step size based on an embedded lower-order estimate.
    E32 = AdaptERK(f, Y0, AdaptERK.ERK32(), rtol, atol, [], [], [], [], [], true);

    [Y_E32, success] = E32.Evolve(tvals, Y0, 0, {w});
    if ~success
        fprintf('  solve failed\n');
    end
    step_hist_E32 = E32.get_step_history();
    err_E32 = norm(Y_E32 - Ytrue, 1);
    fprintf('  steps = %5d  fails = %2d, error = %.2e\n\n', ...
        E32.get_num_steps(), E32.get_num_error_failures(), err_E32);

    if w < 1000
        figure();
        plot(tvals, Y_E32);
        xlabel('t');
        ylabel('y');
        title(sprintf('KPR Solution (w = %d)', w));
        saveas(gcf, sprintf('kpr_solution_w%d.png', w));
    end

    figure(1);
    label = sprintf('w = %d (mean = %.0e)', w, mean(step_hist_E32.h));
    plot(step_hist_E32.t, step_hist_E32.h, 'DisplayName', label);
    hold on;
end

figure(1);
hold off;
xlabel('t');
ylabel('h');
title('Adaptive step histories');
legend('Location', 'best');
saveas(gcf, 'adaptive_steps_w.png');
