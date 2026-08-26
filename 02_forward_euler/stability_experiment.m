% Script to test the forward Euler method for the Dahlquist test problem
%     y' = lambda*y, t in [0,0.5],
%     y(0) = 1,
% for lambda = -100, h in {0.005, 0.01, 0.02, 0.04}
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

% problem time interval and Dahlquist parameter
t0 = 0.0;
tf = 0.4;
lam = -100.0;

% problem-defining functions
ytrue = @(t) exp(lam*t);
f = @(t,y) lam*y;

% shared testing data
Nout = 11;      % includes initial condition
tspan = linspace(t0, tf, Nout).';
hvals = [0.005, 0.01, 0.02, 0.04];

% create true solution results
Ytrue = zeros(Nout,1);
for i = 1:Nout
    Ytrue(i) = ytrue(tspan(i));
end

% loop over time step sizes; call stepper and compute errors
FE = ForwardEuler(f);
for h = hvals

    % set initial condition and call stepper
    y0 = Ytrue(1);
    fprintf('  h = %.6g:\n', h);
    FE.reset();
    [Y, success] = FE.Evolve(tspan, y0, h);

    % output solution, errors, and overall error
    Yerr = abs(Y - Ytrue);
    for i = 1:Nout
        fprintf('    y(%.1f) = %9.6f  \t|error| = %.2e\n', tspan(i), Y(i,1), Yerr(i));
    end
    fprintf('  overall:  steps = %5d  abserr = %9.2e\n\n', FE.get_num_steps(), norm(Yerr, inf));
end
