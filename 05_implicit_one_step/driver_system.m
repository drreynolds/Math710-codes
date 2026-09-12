function driver_system(N)
% Main routine to test the forward Euler method on a system of ODEs
%    y' = f(t,y), t in [0,1],
%    y(0) = y0.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'shared'));

% get optional inputs, otherwise use default values
if nargin < 1 || isempty(N)
    N = 5;
end
fprintf('\nRunning system ODE problem with N = %d\n', N);

% Build a diagonalizable linear test problem with known modal decay rates.
V = eye(N) + rand(N,N);
d = -rand(N,1);
D = diag(d);
% Construct A = V*D*V^{-1} using right division instead of forming inv(V).
A = V * D / V;
if N < 10
    fprintf('\nProblem-defining matrices:\n');
    fprintf('V:\n'); disp(V);
    fprintf('V \\ eye(N):\n'); disp(V \ eye(N));
    fprintf('D:\n'); disp(D);
    fprintf('A:\n'); disp(A);
end

t0 = 0.0;
tf = 1.0;
y0 = rand(N,1);
z0 = V \ y0;

% problem-defining functions
ytrue = @(t) (exp((t(:)-t0) * d(:).') .* z0(:).') * V.';

solver = ImplicitSolver(@(~,~) A, 20, 1e-12, 1e-14, 2);

Nout = 3;
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';
% Compute the analytical solution at the same output times as the methods.
Ytrue = ytrue(tspan);

% set requested time step sizes for convergence tests
hvals = 0.5 ./ linspace(1, 5, 5);

RIIA2 = IRK(@(~,y) A*y, solver, IRK.RadauIIA2());
runTest(RIIA2, 'RadauIIA-2', hvals, Ytrue, tspan);

Alex3 = DIRK(@(~,y) A*y, solver, DIRK.Alexander3());
runTest(Alex3, 'Alexander-3', hvals, Ytrue, tspan);

CR3 = DIRK(@(~,y) A*y, solver, DIRK.CrouzeixRaviart3());
runTest(CR3, 'Crouzeix & Raviart-3', hvals, Ytrue, tspan);

GL2 = IRK(@(~,y) A*y, solver, IRK.GaussLegendre2());
runTest(GL2, 'Gauss-Legendre-2', hvals, Ytrue, tspan);

RIIA3 = IRK(@(~,y) A*y, solver, IRK.RadauIIA3());
runTest(RIIA3, 'RadauIIA-3', hvals, Ytrue, tspan);

GL3 = IRK(@(~,y) A*y, solver, IRK.GaussLegendre3());
runTest(GL3, 'Gauss-Legendre-3', hvals, Ytrue, tspan);

GL6 = IRK(@(~,y) A*y, solver, IRK.GaussLegendre6());
runTest(GL6, 'Gauss-Legendre-6', hvals, Ytrue, tspan);
end

function runTest(stepper, name, hvals, Ytrue, tspan)
    % store errors for convergence-rate estimates
    errs = zeros(size(hvals));
    fprintf('\n%s tests:\n', name);
    for idx = 1:numel(hvals)
        h = hvals(idx);
        fprintf('    h = %.3f:', h);
        stepper.reset();
        stepper.sol.reset();
        [Y, success] = stepper.Evolve(tspan, Ytrue(1,:).', h);
        Yerr = abs(Y - Ytrue);
        errs(idx) = norm(Yerr, inf);
        if success
            fprintf('  solves = %4d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                stepper.get_num_solves(), stepper.sol.get_total_iters(), ...
                stepper.sol.get_total_setups(), errs(idx));
        else
            fprintf('  solve failed\n');
        end
    end
    orders = log(errs(1:end-2)./errs(2:end-1))./log(hvals(1:end-2)./hvals(2:end-1));
    fprintf('    estimated order:  max = %.2f,  avg = %.2f\n', max(orders), mean(orders));
end
