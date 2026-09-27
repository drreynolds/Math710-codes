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
%   G = stiffness at slow time scale (varies)
%   w = variable time-scale separation factor (10)
%
% This script uses adaptive explicit and implicit solvers to assess problem
% stiffness as G is varied.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../shared');
addpath('../04_explicit_one_step');

Tf = 5;
Nt = 50;
tvals = linspace(0, Tf, Nt+1).';
epsilon = 0.5;
w = 10;
% Vary G to study stiffness while keeping the fast oscillation frequency fixed.
Gvals = [-1, -10, -100, -1000, -10000];

% problem-defining functions
r = @(t) 0.5*cos(t);
s = @(t, w) cos(w*t);
rdot = @(t) -0.5*sin(t);
sdot = @(t, w) -w*sin(w*t);
utrue = @(t) sqrt(1 + r(t));
vtrue = @(t, w) sqrt(2 + s(t, w));
ytrue = @(t, w) [utrue(t(:)), vtrue(t(:), w)];
f = @(t, y, G) [G, epsilon; epsilon, -1] * [(-1 + y(1)^2 - r(t)) / (2*y(1)); ...
                                            (-2 + y(2)^2 - s(t, w)) / (2*y(2))] ...
                + [rdot(t)/(2*y(1)); sdot(t, w)/(2*sqrt(2+s(t, w)))];
J = @(t, y, G) [G/2 + (G*(1+r(t))+rdot(t))/(2*y(1)^2), epsilon/2 + epsilon*(2+s(t, w))/(2*y(2)^2); ...
                epsilon/2 + epsilon*(1+r(t))/(2*y(1)^2), -1/2 - (2+s(t, w))/(2*y(2)^2)];

figure(1);

for ig = 1:numel(Gvals)
    G = Gvals(ig);
    fprintf('\nKPR problem with G = %d\n\n', G);

    Y0 = ytrue(0, w);
    % compute and store the analytical solution
    Ytrue = ytrue(tvals, w);

    rtol = 1e-3;
    atol = 1e-11;
    % Build fresh explicit and implicit adaptive steppers for this stiffness value.
    solver = ImplicitSolver(J, 20, 1e-9, 1e-12, 3);
    E32 = AdaptERK(f, Y0, AdaptERK.ERK32(), rtol, atol, [], [], [], [], [], true);
    D32 = AdaptDIRK(f, Y0, solver, AdaptDIRK.ESDIRK32(), rtol, atol, [], [], [], [], [], true);

    fprintf('Adaptive ERK32 solver:\n');
    [Y_E32, success] = E32.Evolve(tvals, Y0, 0, {G});
    if ~success
        fprintf('  solve failed\n');
    end
    step_hist_E32 = E32.get_step_history();
    err_E32 = norm(Y_E32 - Ytrue, 1);
    fprintf('  steps = %5d  fails = %2d, error = %.2e\n', ...
        E32.get_num_steps(), E32.get_num_error_failures(), err_E32);

    fprintf('Adaptive DIRK32 solver:\n');
    [Y_D32, success] = D32.Evolve(tvals, Y0, 0, {G});
    if ~success
        fprintf('  solve failed\n');
    end
    step_hist_D32 = D32.get_step_history();
    err_D32 = norm(Y_D32 - Ytrue, 1);
    fprintf('  steps = %5d  fails = %2d, solves = %5d, error = %.2e\n\n', ...
        D32.get_num_steps(), D32.get_num_error_failures(), D32.get_num_solves(), err_D32);

    figure();
    plot(step_hist_E32.t, step_hist_E32.h, 'r-', 'DisplayName', 'ERK32');
    hold on;
    plot(step_hist_D32.t, step_hist_D32.h, 'b-', 'DisplayName', 'DIRK32');
    idx = step_hist_E32.err > 1.0;
    if any(idx)
        plot(step_hist_E32.t(idx), step_hist_E32.h(idx), 'rx', 'HandleVisibility', 'off');
    end
    idx = step_hist_D32.err > 1.0;
    if any(idx)
        plot(step_hist_D32.t(idx), step_hist_D32.h(idx), 'bx', 'HandleVisibility', 'off');
    end
    hold off;
    xlabel('t');
    ylabel('h');
    title(sprintf('Adaptive step history, G = %d', G));
    legend('Location', 'best');
    saveas(gcf, sprintf('adaptive_steps_G%d.png', G));
end
