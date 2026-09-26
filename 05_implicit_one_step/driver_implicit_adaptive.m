% Script that runs various adaptive implicit methods on the Oregonator problem.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'shared'));

% Initial data and tolerances for the stiff Oregonator kinetics test.
y0 = [5.025e-11; 6e-7; 7.236e-8];
t0 = 0.0;
tf = 360.0;
Nout = 100;
rtol = 1e-6;
atol = 1e-12;

% Set output times for the experiment, while the adaptive methods choose their own steps.
tspan = linspace(t0, tf, Nout+1).';
solver = ImplicitSolver(@J, 20, 1e-9, 1e-12, 3);

% Use a high-accuracy reference solution when the host MATLAB has ode15s available.
Yref = referenceSolution(tspan);

% create adaptive DIRK solvers
AD21 = AdaptDIRK(@f, y0, solver, AdaptDIRK.SDIRK21(), rtol, atol, [], [], [], [], [], true);
AD32 = AdaptDIRK(@f, y0, solver, AdaptDIRK.ESDIRK32(), rtol, atol, [], [], [], [], [], true);
AD43 = AdaptDIRK(@f, y0, solver, AdaptDIRK.ESDIRK43(), rtol, atol, [], [], [], [], [], true);
AD54 = AdaptDIRK(@f, y0, solver, AdaptDIRK.ESDIRK54(), rtol, atol, [], [], [], [], [], true);

% adaptive tests
fprintf('\nAdaptive DIRK21 solver:\n');
[Y_AD21, success] = AD21.Evolve(tspan, y0);
if ~success
    fprintf('  solve failed\n');
end
step_hist_AD21 = AD21.get_step_history();
err_AD21 = norm(Y_AD21 - Yref, 1);
fprintf('  steps = %5d  fails = %2d, error = %.2e\n\n', ...
    AD21.get_num_steps(), AD21.get_num_error_failures(), err_AD21);
solver.reset();

fprintf('\nAdaptive DIRK32 solver:\n');
[Y_AD32, success] = AD32.Evolve(tspan, y0);
if ~success
    fprintf('  solve failed\n');
end
step_hist_AD32 = AD32.get_step_history();
err_AD32 = norm(Y_AD32 - Yref, 1);
fprintf('  steps = %5d  fails = %2d, error = %.2e\n\n', ...
    AD32.get_num_steps(), AD32.get_num_error_failures(), err_AD32);
solver.reset();

fprintf('\nAdaptive DIRK43 solver:\n');
[Y_AD43, success] = AD43.Evolve(tspan, y0);
if ~success
    fprintf('  solve failed\n');
end
step_hist_AD43 = AD43.get_step_history();
err_AD43 = norm(Y_AD43 - Yref, 1);
fprintf('  steps = %5d  fails = %2d, error = %.2e\n\n', ...
    AD43.get_num_steps(), AD43.get_num_error_failures(), err_AD43);
solver.reset();

fprintf('\nAdaptive DIRK54 solver:\n');
[Y_AD54, success] = AD54.Evolve(tspan, y0);
if ~success
    fprintf('  solve failed\n');
end
step_hist_AD54 = AD54.get_step_history();
err_AD54 = norm(Y_AD54 - Yref, 1);
fprintf('  steps = %5d  fails = %2d, error = %.2e\n\n', ...
    AD54.get_num_steps(), AD54.get_num_error_failures(), err_AD54);
solver.reset();

% create plots for adaptive runs
figure();
plot(tspan, Y_AD21);
xlabel('t');
ylabel('y');
title(sprintf('Oregonator Solution (DIRK21, %d steps)', AD21.get_num_steps()));
saveas(gcf, 'adaptive_dirk21.png');

figure();
plot(tspan, Y_AD32);
xlabel('t');
ylabel('y');
title(sprintf('Oregonator Solution (DIRK32, %d steps)', AD32.get_num_steps()));
saveas(gcf, 'adaptive_dirk32.png');

figure();
plot(tspan, Y_AD43);
xlabel('t');
ylabel('y');
title(sprintf('Oregonator Solution (DIRK43, %d steps)', AD43.get_num_steps()));
saveas(gcf, 'adaptive_dirk43.png');

figure();
plot(tspan, Y_AD54);
xlabel('t');
ylabel('y');
title(sprintf('Oregonator Solution (DIRK54, %d steps)', AD54.get_num_steps()));
saveas(gcf, 'adaptive_dirk54.png');

figure();
hold on;
plot(step_hist_AD21.t, step_hist_AD21.h, 'b-', 'DisplayName', 'DIRK21');
plot(step_hist_AD32.t, step_hist_AD32.h, 'k-', 'DisplayName', 'DIRK32');
plot(step_hist_AD43.t, step_hist_AD43.h, 'g-', 'DisplayName', 'DIRK43');
plot(step_hist_AD54.t, step_hist_AD54.h, 'm-', 'DisplayName', 'DIRK54');
for i = 1:numel(step_hist_AD21.t)
    if step_hist_AD21.err(i) > 1.0
        plot(step_hist_AD21.t(i), step_hist_AD21.h(i), 'bx', 'HandleVisibility', 'off');
    end
end
for i = 1:numel(step_hist_AD32.t)
    if step_hist_AD32.err(i) > 1.0
        plot(step_hist_AD32.t(i), step_hist_AD32.h(i), 'kx', 'HandleVisibility', 'off');
    end
end
for i = 1:numel(step_hist_AD43.t)
    if step_hist_AD43.err(i) > 1.0
        plot(step_hist_AD43.t(i), step_hist_AD43.h(i), 'gx', 'HandleVisibility', 'off');
    end
end
for i = 1:numel(step_hist_AD54.t)
    if step_hist_AD54.err(i) > 1.0
        plot(step_hist_AD54.t(i), step_hist_AD54.h(i), 'mx', 'HandleVisibility', 'off');
    end
end
hold off;
xlabel('t');
ylabel('h');
title('Oregonator -- adaptive step history');
legend('Location', 'best');
saveas(gcf, 'adaptive_DIRK_steps.png');

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
