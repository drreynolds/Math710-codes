function run_all_smoke_tests()
% Minimal smoke test runner for the MATLAB branch.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
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
        'Phase 1 numpy demo',           fullfile(rootDir, '01_initial_demo', 'numpy_demo.m'),                         {};
        'Phase 1 plotting demo',        fullfile(rootDir, '01_initial_demo', 'plotting_demo.m'),                      {};
        'Forward Euler basic',          fullfile(rootDir, '02_forward_euler', 'driver_fwd_euler.m'),                  {};
        'Forward Euler adaptive',       fullfile(rootDir, '02_forward_euler', 'driver_adapt_euler.m'),                {};
        'Forward Euler system',         fullfile(rootDir, '02_forward_euler', 'driver_fwd_euler_system.m'),           {};
        'Adaptive Euler system',        fullfile(rootDir, '02_forward_euler', 'driver_adapt_euler_system.m'),         {};
        'Forward Euler stability',      fullfile(rootDir, '02_forward_euler', 'stability_experiment.m'),              {};
        'Simple implicit driver',       fullfile(rootDir, '03_simple_implicit', 'driver.m'),                          {};
        'Explicit one-step fixed',      fullfile(rootDir, '04_explicit_one_step', 'driver_explicit_fixed.m'),         {};
        'Explicit one-step adaptive',   fullfile(rootDir, '04_explicit_one_step', 'driver_explicit_adaptive.m'),      {false};
        'Explicit time-scale adaptive', fullfile(rootDir, '04_explicit_one_step', 'driver_adaptive_timescale.m'),     {true, false};
        'Explicit one-step stability',  fullfile(rootDir, '04_explicit_one_step', 'driver_explicit_stability.m'),     {};
        'Implicit one-step fixed',      fullfile(rootDir, '05_implicit_one_step', 'driver_implicit_fixed.m'),         {};
        'Implicit one-step system',     fullfile(rootDir, '05_implicit_one_step', 'driver_system.m'),                 {};
        'Implicit one-step adaptive',   fullfile(rootDir, '05_implicit_one_step', 'driver_implicit_adaptive.m'),      {true, false};
        'Implicit adaptive stability',  fullfile(rootDir, '05_implicit_one_step', 'driver_adaptive_stability.m'),     {true, false};
        'Reaction-diffusion driver',    fullfile(rootDir, '05_implicit_one_step', 'driver_reaction_diffusion.m'),     {true};
        'Linear multistep driver',      fullfile(rootDir, '06_linear_multistep', 'driver.m'),                         {true};
        'Multirate driver',             fullfile(rootDir, '08_multirate', 'driver_multirate.m'),                      {true};
        'BVP midpoint driver',          fullfile(rootDir, '99_bvp', 'midpoint_driver.m'),                             {[], true};
        'BVP stencil driver',           fullfile(rootDir, '99_bvp', 'stencil_driver.m'),                              {[], true};
        'BVP fd maker demo',            fullfile(rootDir, '99_bvp', 'fd_maker_demo.m'),                               {};
        'BVP Hermite driver',           fullfile(rootDir, '99_bvp', 'hermite_driver.m'),                              {[], true};
        'BVP Lobatto driver',           fullfile(rootDir, '99_bvp', 'lobatto_driver.m'),                              {[], true};
        'Utilities RK stability',        fullfile(rootDir, 'utilities', 'RK_stability_demo.m'),                        {40, false};
        'Utilities LMM stability',       fullfile(rootDir, 'utilities', 'LMM_stability_demo.m'),                       {80, false};
        'Exponential order driver',     fullfile(rootDir, 'exponential', 'driver_exponential_PDEs_Order.m'),          {false, true}
    };

    nTests = size(tests, 1);
    results = cell(nTests, 1);
    timings = zeros(nTests, 1);
    details = cell(nTests, 1);

    fprintf('\nMATLAB smoke tests\n');
    fprintf('==================\n');

    for i = 1:nTests
        testName = tests{i, 1};
        testFile = tests{i, 2};
        testArgs = tests{i, 3};

        if ~isfile(testFile)
            results{i} = 'SKIP';
            details{i} = 'not yet ported';
            fprintf('[SKIP] %s (%s)\n', testName, relativePath(rootDir, testFile));
            continue;
        end

        t0 = tic;
        try
            runMatlabEntry(testFile, testArgs{:});
            timings(i) = toc(t0);
            results{i} = 'PASS';
            details{i} = '';
            fprintf('[PASS] %s (%.3fs)\n', testName, timings(i));
        catch ME
            timings(i) = toc(t0);
            results{i} = 'FAIL';
            details{i} = ME.message;
            fprintf('[FAIL] %s (%.3fs)\n', testName, timings(i));
            fprintf('       %s\n', ME.message);
        end
    end

    % Final summary table.
    fprintf('\nSummary\n');
    fprintf('-------\n');
    fprintf('%-28s %-6s %-8s %s\n', 'Test', 'Result', 'Time(s)', 'Details');
    for i = 1:nTests
        fprintf('%-28s %-6s %-8.3f %s\n', tests{i,1}, results{i}, timings(i), details{i});
    end

    nPass = sum(strcmp(results, 'PASS'));
    nFail = sum(strcmp(results, 'FAIL'));
    nSkip = sum(strcmp(results, 'SKIP'));
    fprintf('\nCounts: PASS=%d  FAIL=%d  SKIP=%d\n', nPass, nFail, nSkip);
end

function runMatlabEntry(testFile, varargin)
%RUNMATLABENTRY Run a script file or call a same-named function file.
    if isFunctionFile(testFile)
        [fileDir, functionName] = fileparts(testFile);
        oldDir = pwd;
        cleanup = onCleanup(@() cd(oldDir));
        cd(fileDir);
        feval(functionName, varargin{:});
    else
        run(testFile);
    end
end

function tf = isFunctionFile(testFile)
%ISFUNCTIONFILE true when first non-comment content starts with function.
    fid = fopen(testFile, 'r');
    if fid < 0
        error('Unable to open test file: %s', testFile);
    end
    cleanup = onCleanup(@() fclose(fid));

    tf = false;
    while true
        line = fgetl(fid);
        if ~ischar(line)
            return;
        end

        line = strtrim(line);
        if isempty(line) || startsWith(line, '%')
            continue;
        end

        tf = startsWith(line, 'function');
        return;
    end
end

function rel = relativePath(rootDir, absPath)
%RELATIVEPATH Return path display relative to rootDir when possible.
    if startsWith(absPath, rootDir)
        rel = absPath(numel(rootDir) + 2:end);
    else
        rel = absPath;
    end
end
