% Main routine to examine the accuracy of fractional-step (operator-splitting)
% methods, using two experiments.
%
% Experiment 1: splitting order versus sub-integrator order.  We apply several
% two-way splittings to the nonlinear Kvaerno Prothero and Robinson problem:
%    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [ r'(t)/(2u) ]
%    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2v) ]
%         = [f1(t,y)]
%           [f2(t,y)]
% where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5, which has analytical
% solution u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).  We split by components,
% with G = -1, e = 0.5 and w = 2, so that the two pieces evolve on similar
% time scales (this is not a multirate test), applying each splitting
% first with highly accurate sub-integrators (many RK4 sub-steps per
% fractional step), and then with one step of forward Euler, Heun, or RK3 per
% fractional step, to show that the observed order is the minimum of the
% splitting order and the sub-integrator order.
%
% Experiment 2: commuting versus non-commuting pieces.  We apply Lie--Trotter
% splitting with highly accurate sub-integrators to the linear problem
%    y' = (J1 + J2) y,  t in [0,1],  y(0) = [1, 1],
% with
%    J1 = [-1  0]     J2 = [-2  eps]
%         [ 0 -2],         [ 0   -1],
% so that ||J1 J2 - J2 J1|| = |eps|.  For eps = 0 the pieces commute and
% Lie--Trotter splitting is exact (up to the sub-integrator error); otherwise
% its error constant scales with the commutator.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../utilities');
addpath('../04_explicit_one_step');

% fractional-step methods to test, from the catalogue in FractionalStep.m
methods = {'Lie-Trotter', FractionalStep.LieTrotter();
           'Lie-Trotter adjoint', FractionalStep.LieTrotterAdjoint();
           'Strang-Marchuk', FractionalStep.StrangMarchuk();
           'OS2(2,2)-1/4', FractionalStep.OS2(0.25);
           'OS2(2,2)-2', FractionalStep.OS2(2.0);
           'Ruth', FractionalStep.Ruth()};

% step sizes to test
Hvals = 0.05 ./ 2.0.^(0:6);


%%%% Experiment 1 %%%%
fprintf('Experiment 1: splitting order versus sub-integrator order\n');

% KPR problem parameters
t0 = 0.0;
tf = 5.0;
G = -1.0;
e = 0.5;
w = 2.0;
tspan = [t0; tf];

% KPR component functions
r = @(t) 0.5*cos(t);
s = @(t) cos(w*t);
rdot = @(t) -0.5*sin(t);
sdot = @(t) -w*sin(w*t);

% KPR true solution
ytrue = @(t) [sqrt(1+r(t)); sqrt(2+s(t))];
y0 = ytrue(t0);
ref = ytrue(tf);

% KPR right-hand side pieces: f1 advances only u, and f2 advances only v
f1 = @(t, y) [G, e; 0, 0] ...
    * [(-1 + y(1)^2 - r(t)) / (2*y(1)); (-2 + y(2)^2 - s(t)) / (2*y(2))] ...
    + [rdot(t) / (2*y(1)); 0];
f2 = @(t, y) [0, 0; e, -1] ...
    * [(-1 + y(1)^2 - r(t)) / (2*y(1)); (-2 + y(2)^2 - s(t)) / (2*y(2))] ...
    + [0; sdot(t) / (2*y(2))];

% sub-integrators: (name, Butcher table, sub-integrator step size); a step
% size of tf-t0 takes a single step per fractional step
subintegrators = {'accurate RK4', ERK.ERK4(), 1e-3;
                  'forward Euler', ERK.ERK1(), tf-t0;
                  'Heun', ERK.Heun(), tf-t0;
                  'RK3', ERK.ERK3(), tf-t0};
for i = 1:size(subintegrators, 1)
    subname = subintegrators{i, 1};
    B = subintegrators{i, 2};
    hsub = subintegrators{i, 3};
    fprintf('\n  sub-integrator = %s:\n', subname);
    for j = 1:size(methods, 1)
        name = methods{j, 1};
        S = methods{j, 2};
        [errs, orders] = RunTest(S, {f1, f2}, {B, B}, hsub, tspan, y0, ref, Hvals);
        fprintf('    %-20s (splitting order %i):  min err = %8.2e,  estimated orders:  %s\n', ...
                name, S.p, errs(end), strjoin(arrayfun(@(q) sprintf('%5.2f', q), orders, 'UniformOutput', false), ' '));
    end
end


%%%% Experiment 2 %%%%
fprintf('\nExperiment 2: commuting versus non-commuting pieces (Lie-Trotter, accurate RK4)\n');

% problem time interval and initial condition
t0 = 0.0;
tf = 1.0;
y0 = [1.0; 1.0];
tspan = [t0; tf];
J1 = [-1.0, 0.0; 0.0, -2.0];

for eps = [0.0, 0.25, 0.5, 1.0, 2.0]
    J2 = [-2.0, eps; 0.0, -1.0];
    f1 = @(t, y) J1 * y;    % first piece of the right-hand side
    f2 = @(t, y) J2 * y;    % second piece of the right-hand side
    ref = expm((tf-t0)*(J1+J2)) * y0;
    commutator = norm(J1*J2 - J2*J1, inf);
    [errs, orders] = RunTest(FractionalStep.LieTrotter(), {f1, f2}, {ERK.ERK4(), ERK.ERK4()}, 1e-3, tspan, y0, ref, Hvals);
    fprintf('  eps = %4.2f:  ||[J1,J2]|| = %4.2f,  err/H at H = %.2e:  %8.2e,  estimated orders:  %s\n', ...
            eps, commutator, Hvals(end), errs(end)/Hvals(end), strjoin(arrayfun(@(q) sprintf('%5.2f', q), orders, 'UniformOutput', false), ' '));
end


% test runner function: returns the errors and the convergence orders estimated
% from each pair of successive step sizes
function [errs, orders] = RunTest(S, f, B, hsub, tspan, y0, ref, Hvals)
    % Runs the fractional-step method with coefficient table S, using an ERK
    % method with Butcher table B{l} and step size hsub for each partition l,
    % with right-hand side functions f{l}, at each step size in Hvals.
    errs = zeros(size(Hvals));
    for idx = 1:numel(Hvals)
        H = Hvals(idx);
        solvers = cell(1, numel(f));
        for l = 1:numel(f)
            solvers{l} = ERK(f{l}, B{l}, hsub);
        end
        stepper = FractionalStep(S, solvers);
        [Y, success] = stepper.Evolve(tspan, y0, H);
        errs(idx) = norm(Y(end,:).' - ref, inf);
    end
    orders = log(errs(1:end-1)./errs(2:end))./log(Hvals(1:end-1)./Hvals(2:end));
end
