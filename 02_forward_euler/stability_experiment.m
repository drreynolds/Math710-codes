% Main routine to test the forward Euler method for Dahlquist test problem
%     y' = lambda*y, t in [0,0.5],
%     y(0) = 1,
% for lambda = -100, h in {0.005, 0.01, 0.02, 0.04}
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
t0 = 0.0;
tf = 0.4;
lam = -100.0;

% problem-defining functions
ytrue = @(t) exp(lam*t);
f = @(t,y) lam*y;

% shared testing data
Nout = 11;
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';
% set requested time step sizes for convergence tests
hvals = [0.005, 0.01, 0.02, 0.04];

% true solution values
% compute and store the analytical solution
Ytrue = zeros(Nout,1);
for i = 1:Nout
    Ytrue(i) = ytrue(tspan(i));
end

% loop over time step sizes
FE = ForwardEuler(f);
for k = 1:numel(hvals)
    h = hvals(k);

    y0 = Ytrue(1);
    fprintf('  h = %.6g:\n', h);
    FE.reset();
    [Y, success] = FE.Evolve(tspan, y0, h);
    if ~success
        fprintf('    solve failed\n');
        continue;
    end

    Yerr = abs(Y(:,1) - Ytrue);
    for i = 1:Nout
        fprintf('    y(%.1f) = %9.6f   |error| = %.2e\n', tspan(i), Y(i,1), Yerr(i));
    end
    fprintf('  overall:  steps = %5d  abserr = %9.2e\n\n', FE.get_num_steps(), norm(Yerr, inf));
end
