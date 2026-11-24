% Main routine to examine the steady states of fractional-step
% (operator-splitting) methods.  We consider the scalar problem
%    y' = -y + 1 = f1(y) + f2(y),   t in [0, Tf],   y(0) = 0,
% with the "decay" piece f1(y) = -y and the "source" piece f2(y) = 1.  Its
% exact solution y(t) = 1 - exp(-t) approaches the steady state y* = 1, at
% which the two pieces balance.  Neither piece is zero there, however, so
% each sub-step of a splitting method moves the solution away from y* = 1,
% and the method instead converges to the fixed point of its own one-step map.
% With exact sub-flows these fixed points are:
%    Lie-Trotter (decay, then source):      H / (1 - exp(-H))   ~ 1 + H/2
%    Lie-Trotter adjoint (source, decay):   H / (exp(H) - 1)    ~ 1 - H/2
%    Strang (half steps of decay):          (H/2) / sinh(H/2)   ~ 1 - H^2/24
%    Strang (half steps of source):         (H/2) coth(H/2)     ~ 1 + H^2/12
%
% We approximate the exact sub-flows by taking many RK4 sub-steps per
% fractional step (the source piece is integrated exactly by any Runge--Kutta
% method, and the RK4 error for the decay piece is negligible here).  The
% script runs each method to Tf = 30, long enough that every computed
% solution has reached its fixed point, prints the steady-state error
% y(Tf) - 1 alongside the predicted value, and estimates the order of the
% steady-state error in H.  It then plots the Lie-Trotter and Strang
% solutions for several step sizes, to show that each levels off at its own,
% incorrect, steady state.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../utilities');
addpath('../04_explicit_one_step');

% problem time interval and initial condition
t0 = 0.0;
tf = 30.0;
y0 = 0.0;
tspan = [t0; tf];

% exact solution and steady state
ytrue = @(t) 1.0 - exp(-t);
ystar = 1.0;

% right-hand side pieces
fdecay = @(t, y) -y;              % decay piece of the right-hand side
fsource = @(t, y) ones(size(y));  % source piece of the right-hand side

% RK4 sub-integrator step size; the sub-flows are then accurate to roughly
% 1e-10, far below the steady-state errors that we wish to observe
hsub = 1e-2;

% fractional-step methods to test: (name, coefficient table, ordering of the
% right-hand side pieces, predicted fixed point as a function of H).  The
% fourth method is Strang--Marchuk splitting with the partitions exchanged,
% so that the source piece takes the two half steps.
methods = {'Lie-Trotter', FractionalStep.LieTrotter(), {fdecay, fsource}, ...
               @(H) H/(1.0 - exp(-H));
           'Lie-Trotter adjoint', FractionalStep.LieTrotterAdjoint(), {fdecay, fsource}, ...
               @(H) H/(exp(H) - 1.0);
           'Strang (decay halves)', FractionalStep.StrangMarchuk(), {fdecay, fsource}, ...
               @(H) (H/2)/sinh(H/2);
           'Strang (source halves)', FractionalStep.StrangMarchuk(), {fsource, fdecay}, ...
               @(H) (H/2)/tanh(H/2)};

% step sizes to test
Hvals = 1.0 ./ 2.0.^(0:4);


%%%% Steady-state errors %%%%
fprintf('Steady-state errors of splitting methods for y'' = -y + 1 (true steady state y* = 1)\n');
for i = 1:size(methods, 1)
    name = methods{i, 1};
    S = methods{i, 2};
    f = methods{i, 3};
    fixed_point = methods{i, 4};
    fprintf('\n  %s (splitting order %i):\n', name, S.p);
    errs = zeros(size(Hvals));
    for idx = 1:numel(Hvals)
        H = Hvals(idx);
        Y = RunSplitting(S, f, H, hsub, tspan, y0);
        errs(idx) = Y(end) - ystar;
        if (idx == 1)
            fprintf('    H = %6.4f:  y(Tf) - 1 = %10.3e,  predicted = %10.3e\n', ...
                    H, errs(idx), fixed_point(H) - ystar);
        else
            order = log(abs(errs(idx-1)/errs(idx)))/log(Hvals(idx-1)/Hvals(idx));
            fprintf('    H = %6.4f:  y(Tf) - 1 = %10.3e,  predicted = %10.3e,  order = %5.2f\n', ...
                    H, errs(idx), fixed_point(H) - ystar, order);
        end
    end
end


%%%% Solution plots %%%%
% output at every outer step, so that each plotted point is a computed solution
Hplot = [1.0, 0.5, 0.25];
tfine = linspace(t0, 8.0, 401);
fig = figure('Position', [100, 100, 1200, 500]);
plotted = [1, 3];
for j = 1:2
    name = methods{plotted(j), 1};
    S = methods{plotted(j), 2};
    f = methods{plotted(j), 3};
    ax = subplot(1, 2, j);  hold(ax, 'on');
    plot(ax, tfine, ytrue(tfine), 'k-', 'DisplayName', 'exact');
    for H = Hplot
        tout = t0 + H*(0:round(8.0/H));
        Y = RunSplitting(S, f, H, hsub, tout, y0);
        plot(ax, tout, Y, 'o-', 'MarkerSize', 4, 'DisplayName', sprintf('H = %g', H));
    end
    xlabel(ax, 't');
    ylabel(ax, 'y');
    title(ax, name);
    grid(ax, 'on');
    legend(ax, 'Location', 'southeast');
end
saveas(fig, 'splitting_steady_state.png');


% test runner function: returns the solution at the times in tspan
function Y = RunSplitting(S, f, H, hsub, tspan, y0)
    % Runs the fractional-step method with coefficient table S, using many RK4
    % sub-steps of size hsub for each partition l, with right-hand side
    % functions f{l}, outer step size H, and initial condition y0.
    solvers = cell(1, numel(f));
    for l = 1:numel(f)
        solvers{l} = ERK(f{l}, ERK.ERK4(), hsub);
    end
    stepper = FractionalStep(S, solvers);
    [Y, success] = stepper.Evolve(tspan, y0, H);
    Y = Y(:,1);
end
