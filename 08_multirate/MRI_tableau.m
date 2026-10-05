function [A, b] = MRI_tableau(C, BF)
    % Usage: [A, b] = MRI_tableau(C, BF)
    %
    %        Inputs:
    %          C is an explicit MRI-GARK coupling table (as in MRI.m), with
    %             fields 'G' (gamma_{ij}^{k} stored as G(k+1,i,j)) and 'c'
    %          BF is the Butcher table of the fast method, with fields 'A', 'b'
    %             and 'c'
    %
    %        Outputs:
    %          A, b are the GARK coefficient blocks and solution weights of the
    %             resulting two-way GARK method (partition 1 = slow, partition
    %             2 = fast), in the format used by GARK_order
    %
% Function to construct the GARK representation of an explicit MRI-GARK
% method, in which each exact fast solve is replaced by one step of a
% Runge--Kutta method.
%
% The GARK blocks (see "The GARK Tableau of an MRI-GARK Method" in the notes),
% for an s-stage MRI-GARK method with coupling coefficients gamma_{ij}^{k},
% slow base method (A, b, c), and fast method (AF, bF, cF) with sF stages, are:
%   A^{SS} = A,  b^S = b;
%   b^F = [dc_2 bF; ... ; dc_s bF];
%   A^{FF}: block (i,lam) = dc_i AF if lam = i, dc_lam 1 bF^T if lam < i;
%   A^{SF}: row i, block lam = dc_lam bF^T for 2 <= lam <= i;
%   A^{FS}: block row i, column j = a_{i-1,j} 1 + sum_k gamma_{ij}^{k} AF CF^k 1,
% where dc_i = c_i - c_{i-1}, CF = diag(cF), and the fast stages are grouped
% by slow stage i = 2,...,s.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
G = C.G;
K = size(G, 1);
[AS, bS, c] = MRI_base_method(C);
s = numel(c);
dc = diff(c);                    % dc(i-1) = c_i - c_{i-1}, i = 2,...,s
AF = BF.A;
bF = BF.b(:);
cF = BF.c(:);
sF = numel(bF);
nF = (s-1)*sF;                   % total number of fast stages
ones_F = ones(sF, 1);

% fast stages within slow stage i (i = 2,...,s) are rows rows(i)
rows = @(i) ((i-2)*sF+1):((i-1)*sF);

% fast-fast block, and fast weights
AFF = zeros(nF, nF);
for i = 2:s
    AFF(rows(i), rows(i)) = dc(i-1)*AF;
    for lam = 2:(i-1)
        AFF(rows(i), rows(lam)) = dc(lam-1)*(ones_F*bF.');
    end
end
bFF = zeros(nF, 1);
for i = 2:s
    bFF(rows(i)) = dc(i-1)*bF;
end

% slow-fast block
ASF = zeros(s, nF);
for i = 2:s
    for lam = 2:i
        ASF(i, rows(lam)) = dc(lam-1)*bF.';
    end
end

% fast-slow block
AFS = zeros(nF, s);
for i = 2:s
    for j = 1:s
        col = AS(i-1, j)*ones_F;
        for k = 1:K
            col = col + G(k,i,j)*(AF*cF.^(k-1));
        end
        AFS(rows(i), j) = col;
    end
end

A = {AS, ASF; AFS, AFF};
b = {bS, bFF};
end
