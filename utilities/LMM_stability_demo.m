function LMM_stability_demo(nthetas, doPlots)
    % MATLAB teaching demo for LMM stability.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % get optional inputs, otherwise use default values
    if nargin < 1 || isempty(nthetas)
        nthetas = 1000;
    end
    if nargin < 2 || isempty(doPlots)
        doPlots = true;
    end

    % set LMM coefficients for each method
    methods = lmmTables();

    % set the thetas resolution
    thetas = linspace(0, 2*pi, nthetas).';

    fprintf('\nLinear multistep stability boundary samples:\n');
    printSample('AB1', thetas, methods.AB1.a, methods.AB1.b);
    printSample('AM2', thetas, methods.AM2.a, methods.AM2.b);
    printSample('BDF2', thetas, methods.BDF2.a, methods.BDF2.b);

    % allow smoke tests to exercise the demo without generating figures
    if ~doPlots
        return;
    end

    % set the bounding box for plots in the complex plane
    box = [-6, 2, -4, 4];
    zoombox = [-1, 1, -3, 3];

    % set the transparency value for filled regions
    alp = 0.2;

    % plot the Adams-Bashforth stability regions, one at a time
    plotFamily(methods, {'AB1','AB2','AB3','AB4','AB5'}, thetas, box, alp, ...
        'Adams-Bashforth Stability Regions (shaded = stable)', 'AB_stability.pdf', 'best');

    % plot the Adams-Moulton stability regions, one at a time
    plotFamily(methods, {'AM1','AM2','AM3','AM4','AM5','AM6'}, thetas, box, alp, ...
        'Adams-Moulton Stability Regions (shaded = stable)', 'AM_stability.pdf', 'best');

    % plot the BDF stability regions, one at a time
    plotFamily(methods, {'BDF1','BDF2','BDF3','BDF4','BDF5','BDF6'}, thetas, box, alp, ...
        'BDF Stability Regions (shaded = unstable)', 'BDF_stability.pdf', 'best');

    % plot a zoomed-in version of BDF stability regions
    plotFamily(methods, {'BDF1','BDF2','BDF3','BDF4','BDF5','BDF6'}, thetas, zoombox, alp, ...
        'Zoom of BDF Stability Regions (shaded = unstable)', 'BDF_stability_zoom.pdf', 'northeastoutside');
end

function methods = lmmTables()
    methods.AB1 = tableEntry([1, -1], [0, 1]);
    methods.AB2 = tableEntry([1, -1], [0, 3, -1]/2);
    methods.AB3 = tableEntry([1, -1], [0, 23, -16, 5]/12);
    methods.AB4 = tableEntry([1, -1], [0, 55, -59, 37, -9]/24);
    methods.AB5 = tableEntry([1, -1], [0, 1901, -2774, 2616, -1274, 251]/720);

    methods.AM1 = tableEntry([1, -1], 1);
    methods.AM2 = tableEntry([1, -1], [1, 1]/2);
    methods.AM3 = tableEntry([1, -1], [5, 8, -1]/12);
    methods.AM4 = tableEntry([1, -1], [9, 19, -5, 1]/24);
    methods.AM5 = tableEntry([1, -1], [251, 646, -264, 106, -19]/720);
    methods.AM6 = tableEntry([1, -1], [475, 1427, -798, 482, -173, 27]/1440);

    methods.BDF1 = tableEntry([1, -1], 1);
    methods.BDF2 = tableEntry([1, -4/3, 1/3], 2/3);
    methods.BDF3 = tableEntry([1, -18/11, 9/11, -2/11], 6/11);
    methods.BDF4 = tableEntry([1, -48/25, 36/25, -16/25, 3/25], 12/25);
    methods.BDF5 = tableEntry([1, -300/137, 300/137, -200/137, 75/137, -12/137], 60/137);
    methods.BDF6 = tableEntry([1, -360/147, 450/147, -400/147, 225/147, -72/147, 10/147], 60/147);
end

function entry = tableEntry(a, b)
    entry.a = a;
    entry.b = b;
end

function printSample(name, thetas, a, b)
    [x, y] = LMM_stability(thetas, a, b);
    fprintf('  %-4s: first = (% .4e,% .4e),  max |eta| = %.4e\n', name, x(1), y(1), max(abs(x + 1i*y)));
end

function plotFamily(methods, names, thetas, box, alp, plotTitle, fileName, legendLocation)
    % create plots for visual diagnostics
    figure();
    hold on;
    colors = {'b', 'r', 'k', 'g', 'm', 'c'};
    for i = 1:numel(names)
        method = methods.(names{i});
        [x, y] = LMM_stability(thetas, method.a, method.b);
        fill(x, y, colors{i}, 'FaceAlpha', alp, 'EdgeColor', 'none');
        plot(x, y, colors{i}, 'DisplayName', sprintf('p=%d', i));
    end
    legend('Location', legendLocation);
    grid on;
    plot([box(1), box(2)], [0, 0], 'k--', 'HandleVisibility', 'off');
    plot([0, 0], [box(3), box(4)], 'k--', 'HandleVisibility', 'off');
    axis equal;
    xlim(box(1:2));
    ylim(box(3:4));
    title(plotTitle);
    saveas(gcf, fileName);
end
