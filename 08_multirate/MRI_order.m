function [pbase, p, failed] = MRI_order(C, maxorder, tol, verbose)
    % Usage: [pbase, p, failed] = MRI_order(C, maxorder, tol, verbose)
    %
    %        Returns the order pbase of the slow base method, and the order p of
    %        the MRI-GARK method with coupling table C, together with the labels
    %        of the lowest-order failing GARK conditions.  Only conditions through
    %        order maxorder (default 4) are checked, so if every one of them holds
    %        then the method has order AT LEAST maxorder; in that case
    %        p = maxorder, and a warning is printed, since a higher maxorder is
    %        needed to determine the order exactly.  tol (default 1e-10) is the
    %        tolerance for considering a residual to be zero, and verbose
    %        (default false) prints the failing conditions if true.
    %
    %        The fast method is the Gauss--Legendre method with enough stages
    %        that its own order conditions hold for every fast tree that appears
    %        in the GARK conditions through order maxorder: in an order-q tree,
    %        each of the at most q-1 fast-slow edges replaces a single weight by
    %        AF CF^k 1, adding at most K-1 vertices to the corresponding fast
    %        tree (where K is the number of Gamma matrices), so a fast method of
    %        order maxorder + (maxorder-1)*(K-1) suffices.
    %
% Function to check the order conditions of explicit MRI-GARK methods, to any
% order, through their GARK representation.  Following the lecture notes, we
% replace each exact fast solve by one step of a Runge--Kutta method of
% sufficiently high order; the resulting two-way GARK method (partition 1 =
% slow, partition 2 = fast) is then checked with GARK_order from
% 07_implicit_explicit/GARK_order.m, which enumerates bicolored rooted trees.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
if nargin < 2 || isempty(maxorder)
    maxorder = 4;
end
if nargin < 3 || isempty(tol)
    tol = 1e-10;
end
if nargin < 4 || isempty(verbose)
    verbose = false;
end

% GARK_order is in the 07_implicit_explicit folder
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), '07_implicit_explicit'));

K = size(C.G, 1);
pF = maxorder + (maxorder-1)*(K-1);
BF = Gauss_tableau(ceil(pF/2));
[A, b] = MRI_tableau(C, BF);
pbase = GARK_order({A{1,1}}, {b{1}}, tol, maxorder);
[p, failed] = GARK_order(A, b, tol, maxorder, verbose);
if (p == maxorder)
    fprintf('  MRI_order warning: all conditions through order %i hold, so the order is at least %i;\n', maxorder, maxorder);
    fprintf('    call with a larger maxorder to determine it exactly\n');
end
end
