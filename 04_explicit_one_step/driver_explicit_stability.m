function driver_explicit_stability()
% Script to test the forward Euler and some fixed-step ERK methods on the
% Dahlquist test problem
%     y' = lambda*y, t in [0,0.5],
%     y(0) = 1,
% for lambda = -100, h in {0.005, 0.01, 0.02, 0.04}
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
t0 = 0.0;
tf = 0.6;
lam = -70.0;

% problem-defining functions
ytrue = @(t) exp(lam*t);
f = @(t, y) lam*y(:);
f_t = @(t, y) 0.0;
f_y = @(t, y) lam;

Nout = 6;    % includes initial condition
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';
% set requested time step sizes for convergence tests
hvals = [0.01, 0.02, 0.03, 0.04];

% Compute the analytical solution at the same output times as the methods.
Ytrue = zeros(Nout, 1);
for i = 1:Nout
    Ytrue(i,:) = ytrue(tspan(i));
end

% Compare stability behavior for methods with progressively larger stable regions.
FE = ERK(f, ERK.ERK1());
fprintf('\nForward Euler:\n');
run_stepper(FE, hvals, Ytrue, tspan);

T2 = Taylor2(f, f_t, f_y);
fprintf('\n2nd order Taylor:\n');
run_stepper(T2, hvals, Ytrue, tspan);

RK4 = ERK(f, ERK.ERK4());
fprintf('\n4th order explicit Runge-Kutta:\n');
run_stepper(RK4, hvals, Ytrue, tspan);
end

function run_stepper(stepper, hvals, Ytrue, tspan)
    % Runs a given stepper on the test problem for a range of time step sizes,
    % and computes the errors.

    for idx = 1:numel(hvals)
        h = hvals(idx);

        % Use the exact initial value so that instability shows up in the time stepper.
        y0 = Ytrue(1,:).';
        fprintf('  h = %.6g:\n', h);
        stepper.reset();
        % Advance with a fixed internal step size h to expose stability limits.
        [Y, success] = stepper.Evolve(tspan, y0, h);
        if ~success
            fprintf('    solve failed at this step size\n');
            continue;
        end

        Yerr = abs(Y - Ytrue);
        % Highlight output values whose error has grown beyond an O(1) threshold.
        for i = 1:numel(tspan)
            text = sprintf('    y(%.3f) = %9.6f   |error| = %.2e', tspan(i), Y(i,1), Yerr(i,1));
            if Yerr(i,1) > 1.0
                fprintf('%s ** failure **\n', text);
            else
                fprintf('%s\n', text);
            end
        end
        fprintf('  overall:  steps = %5d  abserr = %9.2e\n\n', stepper.get_num_steps(), norm(Yerr, inf));
    end
end
