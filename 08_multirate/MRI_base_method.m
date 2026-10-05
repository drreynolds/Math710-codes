function [A, b, c] = MRI_base_method(C)
    % Usage: [A, b, c] = MRI_base_method(C)
    %
    %        Returns the slow base method of the MRI-GARK method with coupling
    %        table C (as in MRI.m), using a_{ij} = sum_{lam<=i} gbar_{lam,j}, with
    %        gbar_{ij} = sum_k gamma_{ij}^{k}/(k+1), and b^T = e_s^T A.
    %
% Function to construct the slow base method of an MRI-GARK method.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
G = C.G;
c = C.c(:);
K = size(G, 1);
s = numel(c);
Gbar = zeros(s, s);
for k = 1:K
    Gbar = Gbar + reshape(G(k,:,:), s, s)/k;
end
A = cumsum(Gbar, 1);
b = A(end,:).';
end
