function [N, hsub] = substeps(dt, h)
    % substeps.m
    %
    % Utility routine shared by the fixed-step solver classes, to decide how
    % many internal steps are required to traverse an output interval.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    % Usage: [N, hsub] = substeps(dt, h)
    %
    % Determines the number of equal internal steps N needed to traverse an
    % output interval of length dt without any step exceeding the requested
    % step size h, and the corresponding step size hsub = dt/N <= h.
    %
    % The small tolerance keeps roundoff in the output times (e.g., from
    % tspan = h*(0:n), where dt/h may be 1 + 1e-16) from adding an extra
    % step; max ensures that we always take at least one step.

    % Fraction of a step by which roundoff in an output interval is forgiven.
    % This must be much larger than roundoff, but much smaller than one step.
    tol = 1e-8;

    N = max(1, ceil(dt/h - tol));
    hsub = dt / N;
end
