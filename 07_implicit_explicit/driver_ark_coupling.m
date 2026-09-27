% Main routine to demonstrate the effect of the ARK coupling conditions,
% using pairs of explicit and implicit Butcher tables that are each
% accurate on their own, applied to the nonstiff additively-split problem
%    y' = fE(t,y) + fI(t,y),  t in [0,2],
%    y(0) = [1, 1/2],
% where
%    fE(t,y) = [cos(t) y_2^2, -sin(t) y_1]   (nonlinear, treated explicitly)
%    fI(t,y) = lambda*y,  lambda = -5          (linear, treated implicitly)
%
% The four pairs are:
%    Heun + implicit midpoint:  each 2nd order, but c^E != c^I, so the
%                               2nd-order coupling conditions fail (order 1)
%    ARS(1,2,2):                explicit + implicit midpoint, shared c (order 2)
%    RK4 + ESDIRK3:             shared c, each at least 3rd order, but the
%                               3rd-order coupling conditions fail (order 2)
%    ERK3 + ESDIRK3:            same ESDIRK and c, with an ERK whose b
%                               matches the ESDIRK (order 3)
%
% Since fI is linear, each implicit stage requires only a linear solve.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
addpath('../shared');

% problem time interval and parameters
t0 = 0.0;
tf = 2.0;
lam = -5.0;
y0 = [1.0; 0.5];

% problem-defining functions
% Explicit (nonstiff) portion of the right-hand side
fE = @(t,y) [cos(t)*y(2)^2; -sin(t)*y(1)];
% Implicit portion of the right-hand side
fI = @(t,y) lam*y;
% Jacobian of the implicit portion of the right-hand side
JI = @(t,y) lam*eye(numel(y));

% reference solution
opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-14);
[~, Yref] = ode45(@(t,y) fE(t,y)+fI(t,y), [t0, tf], y0, opts);
ref = Yref(end,:).';

% test runner function
hvals = 0.1 ./ 2.0.^(0:5);

[BE, BI] = ARK.HeunImplicitMidpoint();
RunTest(BE, BI, 'Heun + implicit midpoint', hvals, ref, t0, tf, y0, fE, fI, JI);
[BE, BI] = ARK.ARS122();
RunTest(BE, BI, 'ARS(1,2,2)', hvals, ref, t0, tf, y0, fE, fI, JI);
[BE, BI] = ARK.RK4ESDIRK3();
RunTest(BE, BI, 'RK4 + ESDIRK3', hvals, ref, t0, tf, y0, fE, fI, JI);
[BE, BI] = ARK.ERK3ESDIRK3();
RunTest(BE, BI, 'ERK3 + ESDIRK3', hvals, ref, t0, tf, y0, fE, fI, JI);

function RunTest(BE, BI, name, hvals, ref, t0, tf, y0, fE, fI, JI)
    errs = zeros(size(hvals));
    fprintf('\n%s tests:\n', name);
    solver = ImplicitSolver(JI, 8, 1e-12, 1e-14);
    stepper = ARK(fE, fI, solver, BE, BI);
    for idx = 1:numel(hvals)
        h = hvals(idx);
        stepper.reset();
        stepper.sol.reset();
        [Y, success] = stepper.Evolve([t0, tf], y0, h);
        errs(idx) = norm(Y(end,:).' - ref, inf);
        if success
            fprintf('    h = %.5f:  solves = %4d  abserr = %8.2e\n', ...
                    h, stepper.get_num_solves(), errs(idx));
        end
    end
    orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
    fprintf('    estimated order:  max = %.2f,  avg = %.2f\n', ...
            max(orders), mean(orders));
end
