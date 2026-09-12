function driver_explicit_fixed()
% Main routine to test the higher-order one-step methods.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
t0 = 0.0;
tf = 1.0;

% problem-defining functions
f = @(t, y) -y(:)*exp(-t);
f_t = @(t, y) y(:)*exp(-t);
f_y = @(t, y) -exp(-t);
ytrue = @(t) exp(exp(-t)-1.0);

% shared testing data
Nout = 3;   % includes initial condition
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';

% Compute the analytical solution at the same output times as the methods.
Ytrue = zeros(Nout, 1);
for i = 1:Nout
    Ytrue(i,:) = ytrue(tspan(i));
end

% time steps to try
% set requested time step sizes for convergence tests
hvals = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005];

% Each call to run_stepper reuses this problem and reports the observed order.
fprintf('\nERK1:\n');
FE = ERK(f, ERK.ERK1());
run_stepper(FE, hvals, Ytrue, tspan);

fprintf('\nTaylor2:\n');
T2 = Taylor2(f, f_t, f_y);
run_stepper(T2, hvals, Ytrue, tspan);

fprintf('\nHeun:\n');
H = ERK(f, ERK.Heun());
run_stepper(H, hvals, Ytrue, tspan);

fprintf('\nERK2:\n');
E2 = ERK(f, ERK.ERK2());
run_stepper(E2, hvals, Ytrue, tspan);

fprintf('\nERK3:\n');
E3 = ERK(f, ERK.ERK3());
run_stepper(E3, hvals, Ytrue, tspan);

fprintf('\nERK4:\n');
E4 = ERK(f, ERK.ERK4());
run_stepper(E4, hvals, Ytrue, tspan);
end


function run_stepper(stepper, hvals, Ytrue, tspan)
    % Runs a given stepper on the test problem for a range of time step sizes,
    % and computes the errors.

    % store errors for convergence-rate estimates
    errs = zeros(size(hvals));
    Nout = numel(tspan);

    for idx = 1:numel(hvals)
        h = hvals(idx);

        % Use the exact initial value so that the measured error is due to time stepping.
        y0 = Ytrue(1,:).';
        fprintf('  h = %.6g:\n', h);
        stepper.reset();
        % Advance to every requested output time using the requested internal step size.
        [Y, success] = stepper.Evolve(tspan, y0, h);
        if ~success
            fprintf('    solve failed at this step size\n');
            continue;
        end

        Yerr = abs(Y - Ytrue);
        errs(idx) = norm(Yerr, inf);
        % Print solution and pointwise errors before summarizing the whole run.
        for i = 1:Nout
            fprintf('    y(%.1f) = %9.6f   |error| = %.2e\n', tspan(i), Y(i,1), Yerr(i,1));
        end
        fprintf('  overall:  steps = %5d  nrhs = %5d  abserr = %9.2e  relerr = %9.2e\n\n', ...
            stepper.get_num_steps(), stepper.get_num_rhs(), errs(idx), norm(Yerr./Ytrue, inf));
    end

    orders = log(errs(1:end-2)./errs(2:end-1))./log(hvals(1:end-2)./hvals(2:end-1));
    fprintf('estimated order: max = %.4f,  avg = %.4f\n', max(orders), mean(orders));
end
