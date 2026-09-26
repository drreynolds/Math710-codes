function [R, Rlim] = ARK_stiff_limit(BE, BI)
    % Usage: [R, Rlim] = ARK_stiff_limit(BE, BI)
    %
    %        Inputs:
    %          BE, BI are the explicit and implicit Butcher tables (see
    %             ARK_order_conditions)
    %
    %        Outputs:
    %          R is the ARK amplification function R(zE,zI) as a sym expression
    %             in the symbols zE and zI, where
    %                R(zE,zI) = 1 + (zE*bE + zI*bI)^T (I - zE*AE - zI*AI)^{-1} 1
    %          Rlim is the limit of R(zE,zI) as zI -> -infinity, as a function of zE
    %
    %        For the combined method to be "L-stable" in the sense of Kennedy &
    %        Carpenter, Rlim should equal zero for every zE.
    %
    % Function to compute the stiff-limit amplification function for two-component
    % additive Runge--Kutta (ARK) methods.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC
    %
    syms zE zI
    AE = sym(BE.A);  bE = sym(BE.b(:));
    AI = sym(BI.A);  bI = sym(BI.b(:));
    s = size(AE, 1);
    M = eye(s) - zE*AE - zI*AI;
    R = simplify(1 + (zE*bE + zI*bI).' * (M \ ones(s, 1)));
    Rlim = simplify(limit(R, zI, -Inf));
end
