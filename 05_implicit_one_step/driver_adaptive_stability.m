function driver_adaptive_stability(quickMode, doPlots)
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
%
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'shared'));
addpath(fullfile(here, '..', '04_explicit_one_step'));

% get optional inputs, otherwise use default values
if nargin < 1 || isempty(quickMode)
    quickMode = false;
end
if nargin < 2 || isempty(doPlots)
    doPlots = true;
end

Tf = 5;
Nt = 50;
tvals = linspace(0, Tf, Nt+1).';
epsilon = 0.5;
w = 10;
% Vary G to study stiffness while keeping the fast oscillation frequency fixed.
Gvals = [-1, -10, -100, -1000, -10000];

if quickMode
    Gvals = [-1, -100, -1000];
end

if doPlots
    % create plots for visual diagnostics
    figure(1);
end

for ig = 1:numel(Gvals)
    G = Gvals(ig);
    fprintf('\nKPR problem with G = %d\n\n', G);

    Y0 = ytrue(0, w);
    % compute and store the analytical solution
    Ytrue = ytrue(tvals, w);

    rtol = 1e-3;
    atol = 1e-11;
    % Build fresh explicit and implicit adaptive steppers for this stiffness value.
    solver = ImplicitSolver(@(t,y) J(t, y, G, epsilon, w), 20, 1e-9, 1e-12, 3);
    E32 = AdaptERK(@(t,y) f(t, y, G, epsilon, w), Y0, AdaptERK.ERK32(), rtol, atol, [], [], [], [], [], true);
    D32 = AdaptDIRK(@(t,y) f(t, y, G, epsilon, w), Y0, solver, AdaptDIRK.ESDIRK32(), rtol, atol, [], [], [], [], [], true);

    fprintf('Adaptive ERK32 solver:\n');
    [Y_E32, success] = E32.Evolve(tvals, Y0);
    if ~success
        fprintf('  solve failed\n');
    end
    step_hist_E32 = E32.get_step_history();
    err_E32 = norm(Y_E32 - Ytrue, 1);
    fprintf('  steps = %5d  fails = %2d, error = %.2e\n', ...
        E32.get_num_steps(), E32.get_num_error_failures(), err_E32);

    fprintf('Adaptive DIRK32 solver:\n');
    [Y_D32, success] = D32.Evolve(tvals, Y0);
    if ~success
        fprintf('  solve failed\n');
    end
    step_hist_D32 = D32.get_step_history();
    err_D32 = norm(Y_D32 - Ytrue, 1);
    fprintf('  steps = %5d  fails = %2d, solves = %5d, error = %.2e\n\n', ...
        D32.get_num_steps(), D32.get_num_error_failures(), D32.get_num_solves(), err_D32);

    if doPlots
        figure();
        plot(step_hist_E32.t, step_hist_E32.h, 'r-', 'DisplayName', 'ERK32');
        hold on;
        plot(step_hist_D32.t, step_hist_D32.h, 'b-', 'DisplayName', 'DIRK32');
        plotFailures(step_hist_E32, 'rx');
        plotFailures(step_hist_D32, 'bx');
        hold off;
        xlabel('t');
        ylabel('h');
        title(sprintf('Adaptive step history, G = %d', G));
        legend('Location', 'best');
        saveas(gcf, sprintf('adaptive_steps_G%d.png', G));
    end
end
end

function val = r(t)
    val = 0.5*cos(t);
end

function val = s(t, w)
    val = cos(w*t);
end

function val = rdot(t)
    val = -0.5*sin(t);
end

function val = sdot(t, w)
    val = -w*sin(w*t);
end

function val = utrue(t)
    val = sqrt(1 + r(t));
end

function val = vtrue(t, w)
    val = sqrt(2 + s(t, w));
end

function val = ytrue(t, w)
    % Return one column vector for scalar t, or one row per time for vector t.
    if isscalar(t)
        val = [utrue(t); vtrue(t, w)];
    else
        val = [utrue(t(:)), vtrue(t(:), w)];
    end
end

function val = f(t, y, G, epsilon, w)
    % Combine the stiff algebraic residual with the forcing that makes ytrue exact.
    u = y(1);
    v = y(2);
    Mat = [G, epsilon; epsilon, -1];
    alg = [(-1 + u^2 - r(t)) / (2*u); ...
           (-2 + v^2 - s(t, w)) / (2*v)];
    forcing = [rdot(t)/(2*u); sdot(t, w)/(2*sqrt(2+s(t, w)))];
    val = Mat*alg + forcing;
end

function val = J(t, y, G, epsilon, w)
    u = y(1);
    v = y(2);
    val = [G/2 + (G*(1+r(t))+rdot(t))/(2*u^2), epsilon/2 + epsilon*(2+s(t, w))/(2*v^2); ...
           epsilon/2 + epsilon*(1+r(t))/(2*u^2), -1/2 - (2+s(t, w))/(2*v^2)];
end

function plotFailures(step_hist, marker)
    % Mark rejected steps on top of each adaptive step-size history.
    idx = step_hist.err > 1.0;
    if any(idx)
        plot(step_hist.t(idx), step_hist.h(idx), marker, 'HandleVisibility', 'off');
    end
end
