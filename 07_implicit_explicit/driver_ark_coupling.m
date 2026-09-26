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
% This driver uses its own small fixed-step ARK routine; once ARK.m is
% added to this folder, that routine should be replaced by the ARK class.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear

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

% reference solution
opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-14);
[~, Yref] = ode45(@(t,y) fE(t,y)+fI(t,y), [t0, tf], y0, opts);
ref = Yref(end,:).';

% test runner function
hvals = 0.1 ./ 2.0.^(0:5);

RunTest(Heun(), ImplicitMidpointPadded(), 'Heun + implicit midpoint', hvals, ref, t0, tf, y0, lam, fE, fI);
RunTest(ExplicitMidpoint(), ImplicitMidpointPadded(), 'ARS(1,2,2)', hvals, ref, t0, tf, y0, lam, fE, fI);
RunTest(RK4(), ESDIRK3(), 'RK4 + ESDIRK3', hvals, ref, t0, tf, y0, lam, fE, fI);
RunTest(ERK3(), ESDIRK3(), 'ERK3 + ESDIRK3', hvals, ref, t0, tf, y0, lam, fE, fI);


% Butcher tables
function B = Heun()
    A = [0.0, 0.0; 1.0, 0.0];
    b = [0.5; 0.5];
    B = struct('A', A, 'b', b, 'c', sum(A,2));
end
function B = ExplicitMidpoint()
    A = [0.0, 0.0; 0.5, 0.0];
    b = [0.0; 1.0];
    B = struct('A', A, 'b', b, 'c', sum(A,2));
end
function B = ImplicitMidpointPadded()
    A = [0.0, 0.0; 0.0, 0.5];
    b = [0.0; 1.0];
    B = struct('A', A, 'b', b, 'c', sum(A,2));
end
function B = RK4()
    A = [0.0, 0.0, 0.0, 0.0; 0.5, 0.0, 0.0, 0.0;
         0.0, 0.5, 0.0, 0.0; 0.0, 0.0, 1.0, 0.0];
    b = [1.0/6.0; 1.0/3.0; 1.0/3.0; 1.0/6.0];
    B = struct('A', A, 'b', b, 'c', sum(A,2));
end
function B = ERK3()
    A = [0.0, 0.0, 0.0, 0.0; 0.5, 0.0, 0.0, 0.0;
         0.0, 0.5, 0.0, 0.0; 1.0, 0.0, 0.0, 0.0];
    b = [1.0/6.0; 0.0; 2.0/3.0; 1.0/6.0];
    B = struct('A', A, 'b', b, 'c', sum(A,2));
end
function B = ESDIRK3()
    A = [0.0, 0.0, 0.0, 0.0; 1.0/6.0, 1.0/3.0, 0.0, 0.0;
         0.5, -1.0/3.0, 1.0/3.0, 0.0; -2.0/3.0, 2.0/3.0, 2.0/3.0, 1.0/3.0];
    b = [1.0/6.0; 0.0; 2.0/3.0; 1.0/6.0];
    B = struct('A', A, 'b', b, 'c', sum(A,2));
end

function y = ark_evolve(BE, BI, h, t0, tf, y0, lam, fE, fI)
    % Usage: y = ark_evolve(BE, BI, h, t0, tf, y0, lam, fE, fI)
    %
    % Fixed-step ARK evolution of the test problem over [t0,tf], using the
    % explicit table BE for fE and the implicit table BI for the linear fI.
    AE = BE.A;  bE = BE.b;  cE = BE.c;
    AI = BI.A;  bI = BI.b;  cI = BI.c;
    s = numel(bE);
    N = round((tf-t0)/h);
    t = t0;
    y = y0;
    for n = 1:N
        FE = {};
        FI = {};
        for i = 1:s
            a = y;
            for j = 1:i-1
                a = a + h*(AE(i,j)*FE{j} + AI(i,j)*FI{j});
            end
            z = a / (1.0 - h*AI(i,i)*lam);
            FE{end+1} = fE(t+cE(i)*h, z);
            FI{end+1} = fI(t+cI(i)*h, z);
        end
        for j = 1:s
            y = y + h*(bE(j)*FE{j} + bI(j)*FI{j});
        end
        t = t + h;
    end
end

function RunTest(BE, BI, name, hvals, ref, t0, tf, y0, lam, fE, fI)
    errs = zeros(size(hvals));
    fprintf('\n%s tests:\n', name);
    for idx = 1:numel(hvals)
        h = hvals(idx);
        errs(idx) = norm(ark_evolve(BE, BI, h, t0, tf, y0, lam, fE, fI) - ref, inf);
        fprintf('    h = %.5f:  abserr = %8.2e\n', h, errs(idx));
    end
    orders = log(errs(1:end-1)./errs(2:end))./log(hvals(1:end-1)./hvals(2:end));
    fprintf('    estimated order:  max = %.2f,  avg = %.2f\n', ...
            max(orders), mean(orders));
end
