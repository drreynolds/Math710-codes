function driver_fwd_euler()
% Main routine to test the forward Euler method on two scalar-valued ODE problems
%    y' = -y, t in [0,5],
%    y(0) = 1.
% and
%    y' = (y+t^2-2)/(t+1), t in [0,5],
%    y(0) = 2.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
t0 = 0.0;
tf = 5.0;

% shared testing data
Nout = 6;   % includes initial condition
% set output times for the experiment
tspan = linspace(t0, tf, Nout);

% create true solution results
Y1true = zeros(Nout,1);
Y2true = zeros(Nout,1);
for i = 1:Nout
    Y1true(i,:) = ytrue1(tspan(i));
    Y2true(i,:) = ytrue2(tspan(i));
end

% time steps to try
% set requested time step sizes for convergence tests
hvals = [0.5, 0.05, 0.005, 0.0005, 0.00005];
% store errors for convergence-rate estimates
errs = zeros(size(hvals));

% problem 1: loop over time step sizes; call stepper and compute errors
fprintf('\nProblem 1:\n');
FE1 = ForwardEuler(@f1);
for idx = 1:numel(hvals)
    h = hvals(idx);

    y0 = Y1true(1,:).';
    fprintf('  h = %.6g:\n', h);
    FE1.reset();
    [Y, success] = FE1.Evolve(tspan, y0, h);
    if ~success
        fprintf('    solve failed at this step size\n');
        continue;
    end

    Yerr = abs(Y - Y1true);
    errs(idx) = norm(Yerr, inf);
    for i = 1:Nout
        fprintf('    y(%.1f) = %9.6f   |error| = %.2e\n', tspan(i), Y(i,1), Yerr(i,1));
    end
    fprintf('  overall:  steps = %5d  abserr = %9.2e\n\n', FE1.get_num_steps(), errs(idx));
end
orders = log(errs(1:end-2)./errs(2:end-1))./log(hvals(1:end-2)./hvals(2:end-1));
fprintf('estimated order: max = %.4f,  avg = %.4f\n', max(orders), mean(orders));


% problem 2: loop over time step sizes; call stepper and compute errors
fprintf('\nProblem 2:\n');
FE2 = ForwardEuler(@f2);
for idx = 1:numel(hvals)
    h = hvals(idx);

    y0 = Y2true(1,:).';
    fprintf('  h = %.6g:\n', h);
    FE2.reset();
    alpha = 1.0;
    beta = 1.0;
    % ForwardEuler expands this cell array as extra RHS arguments, so f2
    % receives alpha and beta after (t,y).
    [Y, success] = FE2.Evolve(tspan, y0, h, {alpha, beta});
    if ~success
        fprintf('    solve failed at this step size\n');
        continue;
    end

    Yerr = abs(Y - Y2true);
    errs(idx) = norm(Yerr, inf);
    for i = 1:Nout
        fprintf('    y(%.1f) = %9.6f   |error| = %.2e\n', tspan(i), Y(i,1), Yerr(i,1));
    end
    fprintf('  overall:  steps = %5d  abserr = %9.2e  relerr = %9.2e\n\n', ...
        FE2.get_num_steps(), errs(idx), norm(Yerr./abs(Y2true), inf));
end
orders = log(errs(1:end-2)./errs(2:end-1))./log(hvals(1:end-2)./hvals(2:end-1));
fprintf('estimated order: max = %.4f,  avg = %.4f\n', max(orders), mean(orders));
end


function val = f1(~, y)
    % ODE RHS function

    val = -y;
end

function val = ytrue1(t)
    % Analytical solution

    val = exp(-t);
end

function val = f2(t, y, alpha, beta)
    % ODE RHS function (with parameters alpha and beta)

    val = (alpha*y(1) + t*t - 2.0*beta)/(t+1.0);
end

function val = ytrue2(t)
    % Analytical solution

    val = t*t + 2.0*t + 2.0 - 2.0*(t+1.0)*log(t+1.0);
end
