function driver_multirate(quickMode)
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
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '04_explicit_one_step'));

% get optional inputs, otherwise use default values
if nargin < 1 || isempty(quickMode)
    quickMode = false;
end

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

if quickMode
    Nt = 5;
    tvals = linspace(0, Tf, Nt+1).';
    w = 50;
    Hvals = [0.1, 0.05];
end

Y0 = ytrue(0, w);
% compute and store the analytical solution
Ytrue = ytrue(tvals, w);

runLT = true;
runSM = true;
runMRI = true;

if runLT
    % Lie-Trotter subcycling evolves slow dynamics once per macro step and
    % fast dynamics with h = H/w substeps.
    runFamily('Lie-Trotter-Subcycling-1', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) LTSubcycling(@(t,y) fs(t,y,G,epsilon,w), ERK.ERK1(), ERK(@(t,y) ff(t,y,G,epsilon,w), ERK.ERK1(), h), H));

    runFamily('Lie-Trotter-Subcycling-2', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) LTSubcycling(@(t,y) fs(t,y,G,epsilon,w), ERK.ERK2(), ERK(@(t,y) ff(t,y,G,epsilon,w), ERK.ERK2(), h), H));
end

if runSM
    % Strang-Marchuk variants symmetrize the slow/fast splitting.
    runFamily('Strang-Marchuk-Subcycling-1', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) SMSubcycling(@(t,y) fs(t,y,G,epsilon,w), ERK.ERK1(), ERK(@(t,y) ff(t,y,G,epsilon,w), ERK.ERK1(), h), H));

    runFamily('Strang-Marchuk-2', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) SMSubcycling(@(t,y) fs(t,y,G,epsilon,w), ERK.ERK2(), ERK(@(t,y) ff(t,y,G,epsilon,w), ERK.ERK2(), h), H));

    runFamily('Strang-Marchuk-3', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) SMSubcycling(@(t,y) fs(t,y,G,epsilon,w), ERK.ERK3(), ERK(@(t,y) ff(t,y,G,epsilon,w), ERK.ERK3(), h), H));
end

if runMRI
    % MRI-GARK methods couple slow stages to a fast IVP solve over each stage interval.
    runFamily('MRI-GARK-ERK22a', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) MRI(Y0, @(t,y) fs(t,y,G,epsilon,w), @(t,y) ff(t,y,G,epsilon,w), MRI.MRIGARKERK22a(), ERK(@(t,y) ff(t,y,G,epsilon,w), ERK.ERK2(), h), H));

    runFamily('MRI-GARK-ERK33a', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) MRI(Y0, @(t,y) fs(t,y,G,epsilon,w), @(t,y) ff(t,y,G,epsilon,w), MRI.MRIGARKERK33a(), ERK(@(t,y) ff(t,y,G,epsilon,w), ERK.ERK3(), h), H));

    runFamily('MRI-GARK-ERK45a', Hvals, w, Y0, Ytrue, tvals, ...
        @(h,H) MRI(Y0, @(t,y) fs(t,y,G,epsilon,w), @(t,y) ff(t,y,G,epsilon,w), MRI.MRIGARKERK45a(), ERK(@(t,y) ff(t,y,G,epsilon,w), ERK.ERK4(), h), H));
end
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

function val = ytrue(t, w)
    if isscalar(t)
        val = [utrue(t); vtrue(t, w)];
    else
        val = [utrue(t(:)), vtrue(t(:), w)];
    end
end

function val = fs(t, y, G, epsilon, w)
    u = y(1);
    v = y(2);
    Mat = [G, epsilon; 0, 0];
    alg = [(-1 + u^2 - r(t)) / (2*u); ...
           (-2 + v^2 - s(t, w)) / (2*v)];
    forcing = [rdot(t)/(2*u); 0];
    val = Mat*alg + forcing;
end

function val = ff(t, y, G, epsilon, w)
    u = y(1);
    v = y(2);
    Mat = [0, 0; epsilon, -1];
    alg = [(-1 + u^2 - r(t)) / (2*u); ...
           (-2 + v^2 - s(t, w)) / (2*v)];
    forcing = [0; sdot(t, w)/(2*v)];
    val = Mat*alg + forcing;
end
