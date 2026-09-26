function [p, failed] = ARK_order(BE, BI, tol, maxorder, verbose)
    % Usage: [p, failed] = ARK_order(BE, BI, tol, maxorder, verbose)
    %
    %        Inputs:
    %          BE, BI are the explicit and implicit Butcher tables (see
    %             ARK_order_conditions)
    %          tol is optional, specifying the tolerance for considering a
    %             residual to be zero (default 1e-10)
    %          maxorder is optional, specifying the highest order to check
    %             (default 4)
    %          verbose is optional, printing the failing conditions if true
    %
    %        Outputs:
    %          p is the order of the ARK method (or maxorder, if all checked
    %             conditions hold)
    %          failed is a cell array of the labels of the lowest-order failing
    %             conditions (empty if p == maxorder)
    %
% Function to determine the order of two-component additive Runge--Kutta (ARK)
% methods, including the coupling conditions.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
if nargin < 3 || isempty(tol)
    tol = 1e-10;
end
if nargin < 4 || isempty(maxorder)
    maxorder = 4;
end
if nargin < 5 || isempty(verbose)
    verbose = false;
end

res = ARK_order_conditions(BE, BI, maxorder);
p = 0;
for q = 1:maxorder
    labels = keys(res{q});
    failed = {};
    for k = 1:numel(labels)
        if (abs(double(vpa(res{q}(labels{k})))) > tol)
            failed{end+1} = labels{k};
        end
    end
    if (numel(failed) > 0)
        if (verbose)
            fprintf('  failing order-%i conditions: %s\n', q, strjoin(failed, ', '));
        end
        return
    end
    p = q;
end
failed = {};
end
