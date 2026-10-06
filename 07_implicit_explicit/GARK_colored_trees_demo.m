% Demo that uses GARK_conditions to print the order conditions for
% generalized-structure additive Runge--Kutta (GARK) methods with an arbitrary
% number M of partitions, and to count the distinct and coupling conditions
% at each order, both as polynomials in M and for specific values of M.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
addpath('../utilities');

% highest order to consider, and specific numbers of partitions to count
maxorder = 6;
Mvals = [2, 3];

cases = {'general', false; 'internally consistent', true};

% print the conditions themselves
for i = 1:size(cases, 1)
    fprintf('\nGARK order conditions (%s), for all colors in {1,...,M}:\n', cases{i, 1});
    for q = 1:maxorder
        fprintf('  order %i:\n', q);
        conds = GARK_conditions(q, cases{i, 2});
        for k = 1:numel(conds)
            fprintf('    %s = %s\n', conds(k).label, char(conds(k).rhs));
        end
    end
end

% count the distinct and coupling conditions as polynomials in M
for i = 1:size(cases, 1)
    fprintf('\nNumber of distinct conditions (%s), as polynomials in M:\n', cases{i, 1});
    for q = 1:maxorder
        fprintf('  order %i:  total = %s,  coupling = %s\n', q, ...
                polynomial_string(count_polynomial(q, cases{i, 2}, false)), ...
                polynomial_string(count_polynomial(q, cases{i, 2}, true)));
    end
end

% count the coupling conditions for specific numbers of partitions
fprintf('\nNumber of coupling conditions at each order:\n');
fprintf('  order');
for M = Mvals
    fprintf('%15s', sprintf('M=%i', M));
end
for M = Mvals
    fprintf('%15s', sprintf('M=%i, int.', M));
end
fprintf('\n');
for q = 1:maxorder
    counts = [];
    for M = Mvals
        [~, coupling] = num_conditions(q, M, false);
        counts(end+1) = coupling;
    end
    for M = Mvals
        [~, coupling] = num_conditions(q, M, true);
        counts(end+1) = coupling;
    end
    fprintf('  %5i', q);
    fprintf('%15i', counts);
    fprintf('\n');
end


function n = num_colorings(t, M, internal, root)
    % Usage: n = num_colorings(t, M, internal, root)
    %
    %        Returns the number of distinct colorings of the tree t using M
    %        colors, i.e., the number of distinct order conditions that t
    %        generates for an M-partition GARK method.  If internal is true, the
    %        (non-root) leaves are not colored.  The input root indicates
    %        whether t is a full tree (true), or a subtree below the root
    %        (false).
    %
if (isempty(t) && internal && (~root))
    n = 1;
    return
end
n = M;
k = 1;
while (k <= numel(t))
    % the children are sorted, so identical children are adjacent
    m = 1;
    while ((k+m <= numel(t)) && isequal(t{k+m}, t{k}))
        m = m + 1;
    end
    nc = num_colorings(t{k}, M, internal, false);
    if (m > nc + m - 1)
        n = 0;
    else
        n = n * nchoosek(nc + m - 1, m);
    end
    k = k + m;
end
end


function [total, coupling] = num_conditions(order, M, internal)
    % Usage: [total, coupling] = num_conditions(order, M, internal)
    %
    %        Returns the total number of distinct order conditions of the given
    %        order for an M-partition GARK method, and the number of these that
    %        are coupling conditions (i.e., that are not an order condition of a
    %        single base method).
    %
total = 0;
coupling = 0;
trees = rooted_trees(order);
for k = 1:numel(trees)
    n = num_colorings(trees{k}, M, internal, true);
    total = total + n;
    coupling = coupling + n - M;
end
end


function coeffs = count_polynomial(order, internal, coupling)
    % Usage: coeffs = count_polynomial(order, internal, coupling)
    %
    %        Returns the coefficients [a_0, a_1, ..., a_order] (as exact sym
    %        values) of the polynomial in M that gives the number of distinct
    %        order conditions (or, if coupling is true, coupling conditions) of
    %        the given order for an M-partition GARK method.  Since this number
    %        is a polynomial of degree at most order, it is found by
    %        interpolating its values at M = 0, 1, ..., order.
    %
Ms = 0:order;
vals = sym(zeros(1, order+1));
for i = 1:order+1
    [total, ncoupling] = num_conditions(order, Ms(i), internal);
    if coupling
        vals(i) = ncoupling;
    else
        vals(i) = total;
    end
end
coeffs = sym(zeros(1, order+1));
for i = 1:order+1
    % Lagrange basis polynomial for node Ms(i), as a coefficient list
    basis = sym(1);
    denom = sym(1);
    for j = 1:order+1
        if (j ~= i)
            basis = [sym(0), basis];
            for k = 1:numel(basis)-1
                basis(k) = basis(k) - Ms(j)*basis(k+1);
            end
            denom = denom * (Ms(i) - Ms(j));
        end
    end
    for k = 1:order+1
        coeffs(k) = coeffs(k) + vals(i)*basis(k)/denom;
    end
end
end


function s = polynomial_string(coeffs)
    % Usage: s = polynomial_string(coeffs)
    %
    %        Returns a string for the polynomial in M with the coefficients
    %        [a_0, a_1, ...], listing the highest powers first.
    %
s = '';
for k = numel(coeffs)-1:-1:0
    a = coeffs(k+1);
    if (a == 0)
        continue
    end
    if (a < 0)
        sgn = ' - ';
    else
        sgn = ' + ';
    end
    if isempty(s)
        if (a < 0)
            sgn = '-';
        else
            sgn = '';
        end
    end
    a = abs(a);
    if (k == 0)
        mono = '';
    elseif (k == 1)
        mono = 'M';
    else
        mono = sprintf('M^%i', k);
    end
    if isempty(mono)
        term = char(a);
    elseif (a == 1)
        term = mono;
    else
        term = [char(a), ' ', mono];
    end
    s = [s, sgn, term];
end
if isempty(s)
    s = '0';
end
end
