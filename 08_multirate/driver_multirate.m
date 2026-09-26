% Script that runs various Lie-Trotter subcycling methods and MRI methods
% on the nonlinear Kvaerno Prothero and Robinson problem:
%    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [ r'(t)/(2u) ]
%    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2v) ]
%         = [fs(t,y)]
%           [ff(t,y)]
% where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5.
% This problem has analytical solution given by
%    u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).
%
% We use the parameters:
%   e = inter-variable coupling strength (0.5)
%   G = stiffness at slow time scale (-10)
%   w = variable time-scale separation factor (100)
%
% This script uses Lie-Trotter subcycling methods with components at various
% orders of accuracy to see if/how that affects accuracy.  It also runs MRI of
% various orders of accuracy to see how those compare.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '04_explicit_one_step'));

Tf = 5;
Nt = 25;
tvals = linspace(0, Tf, Nt+1).';
epsilon = 0.5;
w = 100;
G = -10;
Hvals = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025];

% problem-defining functions
r = @(t) 0.5*cos(t);
s = @(t, w) cos(w*t);
rdot = @(t) -0.5*sin(t);
sdot = @(t, w) -w*sin(w*t);
utrue = @(t) sqrt(1 + r(t));
vtrue = @(t, w) sqrt(2 + s(t, w));
ytrue = @(t) [utrue(t); vtrue(t, w)];

% initial condition
Y0 = ytrue(0);
% compute and store the analytical solution
Ytrue = zeros(Nt+1, 2);
for i = 1:(Nt+1)
    Ytrue(i,:) = ytrue(tvals(i)).';
end

% KPR right-hand side functions
fs = @(t, y) [G, epsilon; 0, 0] ...
    * [(-1 + y(1)^2 - r(t)) / (2*y(1)); (-2 + y(2)^2 - s(t, w)) / (2*y(2))] ...
    + [rdot(t)/(2*y(1)); 0];
ff = @(t, y) [0, 0; epsilon, -1] ...
    * [(-1 + y(1)^2 - r(t)) / (2*y(1)); (-2 + y(2)^2 - s(t, w)) / (2*y(2))] ...
    + [0; sdot(t, w)/(2*y(2))];

runLT = true;
runSM = true;
runMRI = true;

if runLT
    % Lie-Trotter subcycling evolves slow dynamics once per macro step and
    % fast dynamics with h = H/w substeps.
    runFamily('Lie-Trotter-Subcycling-1', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) LTSubcycling(fs, ERK.ERK1(), ERK(ff, ERK.ERK1(), h), H));

    runFamily('Lie-Trotter-Subcycling-2', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) LTSubcycling(fs, ERK.ERK2(), ERK(ff, ERK.ERK2(), h), H));
end

if runSM
    % Strang-Marchuk variants symmetrize the slow/fast splitting.
    runFamily('Strang-Marchuk-Subcycling-1', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) SMSubcycling(fs, ERK.ERK1(), ERK(ff, ERK.ERK1(), h), H));

    runFamily('Strang-Marchuk-2', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) SMSubcycling(fs, ERK.ERK2(), ERK(ff, ERK.ERK2(), h), H));

    runFamily('Strang-Marchuk-3', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) SMSubcycling(fs, ERK.ERK3(), ERK(ff, ERK.ERK3(), h), H));
end

if runMRI
    % MRI-GARK methods couple slow stages to a fast IVP solve over each stage interval.
    runFamily('MRI-GARK-ERK22a', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) MRI(Y0, fs, ff, MRI.MRIGARKERK22a(), ERK(ff, ERK.ERK2(), h), H));

    runFamily('MRI-GARK-ERK33a', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) MRI(Y0, fs, ff, MRI.MRIGARKERK33a(), ERK(ff, ERK.ERK3(), h), H));

    runFamily('MRI-GARK-ERK45a', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) MRI(Y0, fs, ff, MRI.MRIGARKERK45a(), ERK(ff, ERK.ERK4(), h), H));
end

function runFamily(name, Hvals, w, Y0, Ytrue, tvals, buildStepper)
    % store errors for convergence-rate estimates
    errs = zeros(size(Hvals));
    fprintf('\n%s:\n', name);
    for idx = 1:numel(Hvals)
        H = Hvals(idx);
        h = H / w;
        stepper = buildStepper(h, H);
        fast = stepper.FastSolver;
        fprintf('  H = %.6f, h = %.6f:', H, h);
        [Y, success] = stepper.Evolve(tvals, Y0);
        errs(idx) = norm(abs(Y - Ytrue), inf);
        if ~success
            fprintf('  solve failed');
        end
        fprintf('   steps (s,f) = (%d, %d)  nrhs (s,f) = (%d, %d)  err = %.1e\n', ...
            stepper.get_num_steps(), fast.get_num_steps(), stepper.get_num_rhs(), fast.get_num_rhs(), errs(idx));
    end

    if numel(Hvals) > 2
        orders = log(errs(1:end-2)./errs(2:end-1))./log(Hvals(1:end-2)./Hvals(2:end-1));
        fprintf('estimated order: %.2f\n', mean(orders));
    end
end
