function run_all_smoke_tests()
%RUN_ALL_SMOKE_TESTS Minimal smoke test runner for the MATLAB branch.
%   This script is intentionally resilient during incremental porting:
%   missing driver files are reported as SKIP, not failure.

    rootDir = fileparts(mfilename('fullpath'));

    % Keep section ordering aligned with course flow.
    sections = {
        '01_initial_demo', ...
        '02_forward_euler', ...
        '03_simple_implicit', ...
        '04_explicit_one_step', ...
        '05_implicit_one_step', ...
        '06_linear_multistep', ...
        '07_implicit_explicit', ...
        '08_multirate', ...
        '99_bvp', ...
        'exponential', ...
        'shared', ...
        'utilities'
    };

    for i = 1:numel(sections)
        secPath = fullfile(rootDir, sections{i});
        if isfolder(secPath)
            addpath(secPath);
        end
    end

    tests = {
        'Phase 1 numpy demo',           fullfile(rootDir, '01_initial_demo', 'numpy_demo.m');
        'Phase 1 plotting demo',        fullfile(rootDir, '01_initial_demo', 'plotting_demo.m');
        'Forward Euler basic',          fullfile(rootDir, '02_forward_euler', 'driver_fwd_euler.m');
        'Forward Euler adaptive',       fullfile(rootDir, '02_forward_euler', 'driver_adapt_euler.m');
        'Forward Euler system',         fullfile(rootDir, '02_forward_euler', 'driver_fwd_euler_system.m');
        'Adaptive Euler system',        fullfile(rootDir, '02_forward_euler', 'driver_adapt_euler_system.m');
        'Forward Euler stability',      fullfile(rootDir, '02_forward_euler', 'stability_experiment.m');
        'Simple implicit driver',       fullfile(rootDir, '03_simple_implicit', 'driver.m');
        'Explicit one-step fixed',      fullfile(rootDir, '04_explicit_one_step', 'driver_explicit_fixed.m');
        'Implicit one-step fixed',      fullfile(rootDir, '05_implicit_one_step', 'driver_implicit_fixed.m');
        'Linear multistep driver',      fullfile(rootDir, '06_linear_multistep', 'driver.m');
        'Multirate driver',             fullfile(rootDir, '08_multirate', 'driver_multirate.m');
        'BVP midpoint driver',          fullfile(rootDir, '99_bvp', 'midpoint_driver.m');
        'Exponential order driver',     fullfile(rootDir, 'exponential', 'driver_exponential_PDEs_Order.m')
    };

    nTests = size(tests, 1);
    results = strings(nTests, 1);
    timings = zeros(nTests, 1);
    details = strings(nTests, 1);

    fprintf('\nMATLAB smoke tests\n');
    fprintf('==================\n');

    for i = 1:nTests
        testName = tests{i, 1};
        testFile = tests{i, 2};

        if ~isfile(testFile)
            results(i) = "SKIP";
            details(i) = "not yet ported";
            fprintf('[SKIP] %s (%s)\n', testName, relativePath(rootDir, testFile));
            continue;
        end

        t0 = tic;
        try
            run(testFile);
            timings(i) = toc(t0);
            results(i) = "PASS";
            details(i) = "";
            fprintf('[PASS] %s (%.3fs)\n', testName, timings(i));
        catch ME
            timings(i) = toc(t0);
            results(i) = "FAIL";
            details(i) = string(ME.message);
            fprintf('[FAIL] %s (%.3fs)\n', testName, timings(i));
            fprintf('       %s\n', ME.message);
        end
    end

    % Final summary table.
    fprintf('\nSummary\n');
    fprintf('-------\n');
    fprintf('%-28s %-6s %-8s %s\n', 'Test', 'Result', 'Time(s)', 'Details');
    for i = 1:nTests
        fprintf('%-28s %-6s %-8.3f %s\n', tests{i,1}, results(i), timings(i), details(i));
    end

    nPass = sum(results == "PASS");
    nFail = sum(results == "FAIL");
    nSkip = sum(results == "SKIP");
    fprintf('\nCounts: PASS=%d  FAIL=%d  SKIP=%d\n', nPass, nFail, nSkip);
end

function rel = relativePath(rootDir, absPath)
%RELATIVEPATH Return path display relative to rootDir when possible.
    if startsWith(absPath, rootDir)
        rel = absPath(numel(rootDir) + 2:end);
    else
        rel = absPath;
    end
end
