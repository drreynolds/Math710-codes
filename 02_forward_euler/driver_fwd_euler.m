% Script to test the forward Euler method on two scalar-valued ODE problems
%    y' = -y, t in [0,5],
%    y(0) = 1.
% and
%    y' = (y+t^2-2)/(t+1), t in [0,5],
%    y(0) = 2.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
clear
addpath('../utilities');

% problem time interval
t0 = 0.0;
tf = 5.0;

% problem-definining functions
%   ODE RHS function
f1 = @(t,y) -y;
%   Analytical solution
ytrue1 = @(t) exp(-t);

%   ODE RHS function (with parameters)
f2 = @(t,y,alpha,beta) (alpha*y + t*t - 2.0*beta)/(t+1);
%   Analytical solution
ytrue2 = @(t) t*t + 2.0*t + 2.0 - 2.0*(t+1.0)*log(t+1.0);

% shared testing data
Nout = 6;   % includes initial condition
tspan = linspace(t0, tf, Nout);

% create true solution results
Y1true = zeros(Nout,1);
Y2true = zeros(Nout,1);
for i = 1:Nout
    Y1true(i,:) = ytrue1(tspan(i));
    Y2true(i,:) = ytrue2(tspan(i));
end

% time steps to try
hvals = [0.5, 0.05, 0.005, 0.0005, 0.00005];
errs = zeros(size(hvals));

% problem 1: loop over time step sizes; call stepper and compute errors
fprintf('\nProblem 1:\n');
FE1 = ForwardEuler(f1);
for idx = 1:numel(hvals)
    h = hvals(idx);

    % set initial condition and call stepper
    y0 = Y1true(1,:).';
    fprintf('  h = %.6g:\n', h);
    FE1.reset();
    [Y, success] = FE1.Evolve(tspan, y0, h);

    % output solution, errors, and overall error
    Yerr = abs(Y - Y1true);
    errs(idx) = norm(Yerr, inf);
    for i = 1:Nout
        fprintf('    y(%.1f) = %9.6f   |error| = %.2e\n', tspan(i), Y(i,1), Yerr(i,1));
    end
    fprintf('  overall:  steps = %5d  abserr = %9.2e\n\n', FE1.get_num_steps(), errs(idx));
end
orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
fprintf('estimated order: max = %.4f,  avg = %.4f\n', max(orders), mean(orders));


% problem 2: loop over time step sizes; call stepper and compute errors
fprintf('\nProblem 2:\n');
FE2 = ForwardEuler(f2);
for idx = 1:numel(hvals)
    h = hvals(idx);

    % set initial condition and call stepper
    y0 = Y2true(1,:).';
    fprintf('  h = %.6g:\n', h);
    FE2.reset();
    alpha = 1.0;
    beta = 1.0;
    % Here when calling our rhs (f2) we have two parameters alpha and beta.
    % We must pack these into a cell array; the solver will unpack these to provide to f2 following t and y.
    [Y, success] = FE2.Evolve(tspan, y0, h, {alpha, beta});

    % output solution, errors, and overall error
    Yerr = abs(Y - Y2true);
    errs(idx) = norm(Yerr, inf);
    for i = 1:Nout
        fprintf('    y(%.1f) = %9.6f   |error| = %.2e\n', tspan(i), Y(i,1), Yerr(i,1));
    end
    fprintf('  overall:  steps = %5d  abserr = %9.2e  relerr = %9.2e\n\n', ...
        FE2.get_num_steps(), errs(idx), norm(Yerr./abs(Y2true), inf));
end
orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
fprintf('estimated order: max = %.4f,  avg = %.4f\n', max(orders), mean(orders));

