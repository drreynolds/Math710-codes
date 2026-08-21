function driver_implicit_adaptive(quickMode, doPlots)
% Script that runs various adaptive implicit methods on the Oregonator problem.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'shared'));

% get optional inputs, otherwise use default values
if nargin < 1 || isempty(quickMode)
    quickMode = false;
end
if nargin < 2 || isempty(doPlots)
    doPlots = true;
end

% Initial data and tolerances for the stiff Oregonator kinetics test.
y0 = [5.025e-11; 6e-7; 7.236e-8];
t0 = 0.0;
tf = 360.0;
Nout = 100;
rtol = 1e-6;
atol = 1e-12;

if quickMode
    Nout = 24;
    rtol = 1e-5;
end

% Set output times for the experiment, while the adaptive methods choose their own steps.
tspan = linspace(t0, tf, Nout+1).';
solver = ImplicitSolver(@J, 20, 1e-9, 1e-12, 3);

% Compare several embedded DIRK pairs with increasing formal order.
methods = {
    'DIRK21', AdaptDIRK.SDIRK21();
    'DIRK32', AdaptDIRK.ESDIRK32();
    'DIRK43', AdaptDIRK.ESDIRK43();
    'DIRK54', AdaptDIRK.ESDIRK54()
};

% Use a high-accuracy reference solution when the host MATLAB has ode15s available.
Yref = referenceSolution(tspan);
solutions = cell(size(methods,1), 1);
histories = cell(size(methods,1), 1);

for imethod = 1:size(methods,1)
    name = methods{imethod,1};
    B = methods{imethod,2};
    % Reuse the same nonlinear solver settings while swapping the DIRK tableau.
    stepper = AdaptDIRK(@f, y0, solver, B, rtol, atol, [], [], [], [], [], true);

    fprintf('\nAdaptive %s solver:\n', name);
    [Y, success] = stepper.Evolve(tspan, y0);
    if ~success
        fprintf('  solve failed\n');
    end
    histories{imethod} = stepper.get_step_history();
    solutions{imethod} = Y;
    err = norm(Y - Yref, 1);
    fprintf('  steps = %5d  fails = %2d, solves = %5d, error = %.2e\n\n', ...
        stepper.get_num_steps(), stepper.get_num_error_failures(), stepper.get_num_solves(), err);
    solver.reset();
end

% allow smoke tests to exercise the demo without generating figures
if ~doPlots
    return;
end

for imethod = 1:size(methods,1)
    name = methods{imethod,1};
    Y = solutions{imethod};
    % create plots for visual diagnostics
    figure();
    plot(tspan, Y);
    xlabel('t');
    ylabel('y');
    title(sprintf('Oregonator Solution (%s, %d outputs)', name, numel(tspan)));
    saveas(gcf, sprintf('adaptive_%s.png', lower(name)));
end

figure();
hold on;
for imethod = 1:size(methods,1)
    hist = histories{imethod};
    plot(hist.t, hist.h, 'DisplayName', methods{imethod,1});
    plotFailures(hist);
end
hold off;
xlabel('t');
ylabel('h');
title('Oregonator adaptive step history');
legend('Location', 'best');
saveas(gcf, 'adaptive_DIRK_steps.png');
end

function Yref = referenceSolution(tspan)
    % ode15s is used only as an external reference, not as part of the method under test.
    if exist('ode15s', 'file') && ~isOctave()
        opts = odeset('RelTol', 1e-12, 'AbsTol', [1e-16, 1e-20, 1e-18], 'Jacobian', @J);
        try
            [~, Yref] = ode15s(@f, tspan, [5.025e-11; 6e-7; 7.236e-8], opts);
        catch ME
            warning('driver_implicit_adaptive:reference', ...
                'Reference solve failed: %s. Reporting NaN errors.', ME.message);
            Yref = nan(numel(tspan), 3);
        end
    else
        Yref = nan(numel(tspan), 3);
    end
end

function tf = isOctave()
    persistent cached
    if isempty(cached)
        cached = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    end
    tf = cached;
end

function val = f(~, y)
    % Right-hand side function, f(t,y), for the IVP.

    k1 = 2.57555802e8;
    k2 = 7.72667406e1;
    k3 = 1.28777901e7;
    k4 = 1.29421790e-2;
    k5 = 1.60972376e-1;
    val = [-k1*y(1)*y(2) + k2*y(1) - k3*y(1)^2 + k4*y(2); ...
           -k1*y(1)*y(2) - k4*y(2) + k5*y(3); ...
           k2*y(1) - k5*y(3)];
end

function val = J(~, y)
    % Jacobian (in dense matrix format) of the right-hand side
    % function, J(t,y) = df/dy, for the IVP.

    k1 = 2.57555802e8;
    k2 = 7.72667406e1;
    k3 = 1.28777901e7;
    k4 = 1.29421790e-2;
    k5 = 1.60972376e-1;
    val = [-k1*y(2) + k2 - 2.0*k3*y(1), -k1*y(1) + k4, 0; ...
           -k1*y(2), -k1*y(1) - k4, k5; ...
           k2, 0, -k5];
end

function plotFailures(step_hist)
    % Mark rejected steps on the adaptive step-size history.
    idx = step_hist.err > 1.0;
    if any(idx)
        plot(step_hist.t(idx), step_hist.h(idx), 'x', 'HandleVisibility', 'off');
    end
end
