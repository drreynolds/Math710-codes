function [A, b] = FSRK_tableau(alpha, B)
    % Usage: [A, b] = FSRK_tableau(alpha, B)
    %
    %        Inputs:
    %          alpha is a cell array of M vectors, where alpha{l} holds the
    %             fractions alpha_k^{l}, k=1,...,s, of the step taken by
    %             partition l in each stage of the fractional-step method
    %          B is a cell array of M Butcher tables, where B{l} is the
    %             sub-integrator used for every sub-step of partition l, with
    %             components:
    %             B{l}.A -- the Butcher table matrix
    %             B{l}.b -- the solution coefficients
    %
    %        Outputs:
    %          A, b are the GARK coefficient blocks and solution weights of the
    %             resulting fractional-step Runge--Kutta method (see
    %             GARK_order_conditions)
    %
    %        The sub-steps are applied in the order (1,1),(1,2),...,(1,M),
    %        (2,1),...,(s,M), i.e., within each stage partition 1 is advanced
    %        first.
    %
% Function to construct the GARK tableau of a fractional-step Runge--Kutta
% (FSRK) method.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
M = numel(alpha);
s = numel(alpha{1});
Bt = cell(1, M);
st = zeros(1, M);
for l = 1:M
    Bt{l}.A = sym(B{l}.A);
    Bt{l}.b = sym(B{l}.b(:));
    st(l) = size(Bt{l}.A, 1);
end
order = zeros(s*M, 2);
for k = 1:s
    for l = 1:M
        order((k-1)*M+l, :) = [k, l];
    end
end
A = cell(M, M);
for l = 1:M
    for lp = 1:M
        A{l,lp} = sym(zeros(s*st(l), s*st(lp)));
    end
end
for i = 1:size(order, 1)
    k = order(i, 1);
    l = order(i, 2);
    for j = 1:size(order, 1)
        kp = order(j, 1);
        lp = order(j, 2);
        if (j < i)
            block = alpha{lp}(kp)*sym(ones(st(l), 1))*Bt{lp}.b.';
        elseif (j == i)
            block = alpha{l}(k)*Bt{l}.A;
        else
            continue
        end
        A{l,lp}((k-1)*st(l)+1:k*st(l), (kp-1)*st(lp)+1:kp*st(lp)) = block;
    end
end
b = cell(1, M);
for l = 1:M
    b{l} = sym(zeros(s*st(l), 1));
    for k = 1:s
        b{l}((k-1)*st(l)+1:k*st(l)) = alpha{l}(k)*Bt{l}.b;
    end
end
end
