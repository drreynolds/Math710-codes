% Main routine to demonstrate order reduction of fixed-step ARK methods on
% the split Prothero--Robinson problem
%
%   y' = fE(t,y) + fI(t,y),
%   fE(t,y) = mu*(y-phi(t)),                  (nonstiff, treated explicitly)
%   fI(t,y) = lambda*(y-phi(t)) + phi'(t),    (stiff, treated implicitly)
%   phi(t) = sin(t+pi/4),  mu = -1,  t in [0,10],
%
% that has analytical solution y(t) = phi(t).  Since fE vanishes on the true
% solution, the explicit table does not introduce stage errors of its own, so
% the stiff convergence rates are governed by the stage order of the implicit
% table.  The ARS(3,4,3) implicit table has stage order one, whereas the
% implicit table in ARK3(2)4L[2]SA has stage order two.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear
addpath('../shared');

% problem time interval and parameters
t0 = 0.0;
tf = 10.0;
mu = -1.0;

% problem-defining functions
% true solution to the IVP
ytrue = @(t) sin(t + pi/4.0);
% explicit portion of the split Prothero--Robinson problem
fE = @(t, y, lam) mu*(y(1) - ytrue(t));
% implicit portion of the split Prothero--Robinson problem
fI = @(t, y, lam) lam*(y(1) - ytrue(t)) + cos(t + pi/4.0);
% Jacobian of the implicit portion of the right-hand side
JI = @(t, y, lam) lam;

% shared testing data
y0 = ytrue(t0);
tspan = [t0; tf];
lambdas = -10.0.^(1:4);
hvals = tf ./ (10 * 2.^(0:2:10));

[BE, BI] = ARK.ARS343();
RunTest(BE, BI, 'ARS(3,4,3)', 1, fE, fI, JI, lambdas, hvals, y0, ytrue, tspan, tf);
[BE, BI] = ARK.ARK324L2SA();
RunTest(BE, BI, 'ARK324L2SA', 2, fE, fI, JI, lambdas, hvals, y0, ytrue, tspan, tf);

% test runner function
function RunTest(BE, BI, name, stage_order, fE, fI, JI, lambdas, hvals, y0, ytrue, tspan, tf)

    fprintf('\n%s tests:\n', name);
    figure('Position', [100, 100, 600, 550]);
    colors = lines(numel(lambdas));
    solver = ImplicitSolver(JI, 8, 1e-12, 1e-14, 1);
    stepper = ARK(fE, fI, solver, BE, BI);
    for ilam = 1:numel(lambdas)
        lam = lambdas(ilam);
        errs = zeros(size(hvals));
        fprintf('  lambda = %.1f:\n', lam);
        for idx = 1:numel(hvals)
            h = hvals(idx);
            fprintf('    h = %.5e:', h);
            stepper.reset();
            stepper.sol.reset();
            [Y, success] = stepper.Evolve(tspan, y0, h, {lam});
            errs(idx) = norm(Y(end,:).' - ytrue(tf), inf);
            if success
                fprintf('  solves = %5d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                    stepper.get_num_solves(), stepper.sol.get_total_iters(), ...
                    stepper.sol.get_total_setups(), errs(idx));
            else
                fprintf('  solve failed  abserr = %8.2e\n', errs(idx));
            end
        end
        loglog(hvals, errs, '-o', 'Color', colors(ilam,:), ...
            'MarkerSize', 4, 'DisplayName', sprintf('\\lambda = -10^{%i}', ilam));
        hold on;
    end

    addTrendLines(gca, hvals, 3);
    xlabel('$h$', 'Interpreter', 'latex');
    ylabel('error');
    title(sprintf('%s (order 3, stage order %i)', name, stage_order));
    grid on;
    legend('Location', 'best');
    hold off;
    filename = ['ark_order_reduction_', strrep(strrep(strrep(name, '(', ''), ')', ''), ',', ''), '.png'];
    exportgraphics(gcf, filename);
    fprintf('  saved %s\n', filename);
end

% Short utility function to add log-log reference lines for slopes 1 through order.
function addTrendLines(ax, hvals, order)
    htrend = [hvals(end), hvals(end-2)];
    hratio = htrend(2) / htrend(1);
    ylimits = ylim(ax);
    logymin = log10(ylimits(1));
    logymax = log10(ylimits(2));
    upperStart = logymax - order*log10(hratio);
    logystarts = linspace(logymin + 0.1, upperStart - 0.1, order);
    trendColors = hsv(order);

    for slope = 1:order
        ytrend = 10.0^logystarts(slope) * (htrend / htrend(1)).^slope;
        loglog(ax, htrend, ytrend, '--', 'Color', trendColors(slope,:), ...
            'LineWidth', 1.5, 'DisplayName', sprintf('slope %i', slope));
    end
end
