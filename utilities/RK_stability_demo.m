% MATLAB teaching demo for RK stability.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '04_explicit_one_step'));
addpath(fullfile(here, '..', '05_implicit_one_step'));

% get the plot resolution from the command line, otherwise set to 100
N = str2double(input('Enter the plot resolution N >= 2 [default 100]: ', 's'));
if ~isfinite(N) || ~isreal(N) || N ~= round(N) || N < 2
    fprintf('Invalid or missing N, using the default value 100\n');
    N = 100;
end

methods = {
    'Forward Euler',        forwardEulerTable(),          [-3.0, 1.0, -2.0, 2.0],    'FE_stability.png';
    'ERK4',                 ERK.ERK4(),                   [-5.0, 1.0, -3.0, 3.0],    'RK4_stability.png';
    'Backward Euler',       backwardEulerTable(),         [-1.0, 3.0, -2.0, 2.0],    'BE_stability.png';
    'CrouzeixRaviart3',     DIRK.CrouzeixRaviart3(),      [-10.0, 10.0, -10.0, 10.0], 'CR3_stability.png'
};

fprintf('\nRunge-Kutta stability boundary samples:\n');
for i = 1:size(methods, 1)
    name = methods{i, 1};
    B = methods{i, 2};
    box = methods{i, 3};
    fileName = methods{i, 4};
    [x, y, R] = RK_stability(B, box, N);
    fprintf('  %-16s: boundary points = %4d,  sampled min/max |R| = %.4e / %.4e\n', ...
        name, numel(x), min(R(:)), max(R(:)));
    plotStabilityRegion(box, R, N, name, fileName, x, y);
end

function B = forwardEulerTable()
    B.A = 0.0;
    B.b = 1.0;
    B.c = 0.0;
    B.p = 1;
end

function B = backwardEulerTable()
    B.A = 1.0;
    B.b = 1.0;
    B.c = 1.0;
    B.p = 1;
end

function plotStabilityRegion(box, R, N, methodName, fileName, xBoundary, yBoundary)
    x = linspace(box(1), box(2), N);
    y = linspace(box(3), box(4), N);
    % create plots for visual diagnostics
    figure();
    hold on;
    contourf(x, y, R, [0.0, 1.0]);
    plot(xBoundary, yBoundary, 'k-', 'LineWidth', 1.25);
    plot([box(1), box(2)], [0, 0], 'k--');
    plot([0, 0], [box(3), box(4)], 'k--');
    grid on;
    axis equal;
    xlim(box(1:2));
    ylim(box(3:4));
    title(sprintf('%s stability region (shaded = stable)', methodName));
    saveas(gcf, fileName);
end
