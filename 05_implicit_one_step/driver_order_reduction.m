% Demonstrate order reduction of fixed-step DIRK methods on the linear
% Prothero--Robinson ODE from Ketcheson, Seibold, Shirokoff, and Zhou (2020),
% Sect. 4.1:
%
%   u' = lambda*(u - phi(t)) + phi'(t),  phi(t) = sin(t + pi/4),  t in [0,10].
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
clear

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'shared'));

% problem time interval and parameters
t0 = 0.0;
tf = 10.0;

% problem-defining functions
ytrue = @(t) sin(t + pi/4.0);
f = @(t, y, lam) lam*(y(1) - ytrue(t)) + cos(t + pi/4.0);
J = @(t, y, lam) lam;

% shared testing data.  Each h is tf/N for an integer N, so the DIRK solver
% uses exactly the h value plotted below.
y0 = ytrue(t0);
tspan = [t0; tf];
lambdas = -10.0.^(1:4);
hvals = tf ./ (10 * 2.^(0:2:10));

% Shared nonlinear solver; runTest resets its statistics before each solve.
solver = ImplicitSolver(J, 8, 1e-12, 1e-14, 1);

% The first three are conventional order >= 3 DIRK methods; the final three
% are the high-WSO methods published in Sect. 3 of the paper.
Alex3 = DIRK(f, solver, DIRK.Alexander3());
SD45 = DIRK(f, solver, DIRK.SDIRK45L1SA());
C6 = DIRK(f, solver, DIRK.Cooper6ESDIRK());
D32 = DIRK(f, solver, DIRK.WSO32());
D33 = DIRK(f, solver, DIRK.WSO33());
D43 = DIRK(f, solver, DIRK.WSO43());

runTest(Alex3, 'Alexander3', 3, 1, lambdas, hvals, y0, ytrue, tspan, tf);
runTest(SD45, 'SDIRK45L1SA', 4, 1, lambdas, hvals, y0, ytrue, tspan, tf);
runTest(C6, 'Cooper6ESDIRK', 5, 1, lambdas, hvals, y0, ytrue, tspan, tf);
runTest(D32, 'WSO32', 3, 2, lambdas, hvals, y0, ytrue, tspan, tf);
runTest(D33, 'WSO33', 3, 3, lambdas, hvals, y0, ytrue, tspan, tf);
runTest(D43, 'WSO43', 4, 3, lambdas, hvals, y0, ytrue, tspan, tf);

function runTest(stepper, name, order, wso, lambdas, hvals, y0, ytrue, tspan, tf)
    % Run all stiffness values for one DIRK method and save its convergence plot.

    fprintf('\n%s tests:\n', name);
    figure;
    colors = lines(numel(lambdas));

    for ilam = 1:numel(lambdas)

        lam = lambdas(ilam);
        errs = zeros(size(hvals));
        fprintf('  lambda = %.1f:\n', lam);
        for idx = 1:numel(hvals)
            h = hvals(idx);
            fprintf('    h = %.5e:', h);
            stepper.reset();
            stepper.sol.reset();
            % Pass lambda through the args cell array to both f and J.
            [Y, success] = stepper.Evolve(tspan, y0, h, {lam});
            if ~success
                error('DIRK solve failed for lambda=%g, h=%g', lam, h);
            end
            errs(idx) = norm(Y(end,:).' - ytrue(tf), inf);
            fprintf('  solves = %5d  Niters = %6d  NJevals = %5d  abserr = %8.2e\n', ...
                stepper.get_num_solves(), stepper.sol.get_total_iters(), ...
                stepper.sol.get_total_setups(), errs(idx));

        end
        loglog(hvals, errs, '-o', 'Color', colors(ilam,:), ...
            'MarkerSize', 4, 'DisplayName', sprintf('\\lambda = -10^{%i}', ilam));
        hold on;
    end

    addTrendLines(gca, hvals, order);
    xlabel('$h$', 'Interpreter', 'latex');
    ylabel('error');
    title(sprintf('%s (order %i, WSO %i)', name, order, wso));
    grid on;
    legend('Location', 'best');
    hold off;

    filename = ['order_reduction_', name, '.png'];
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
