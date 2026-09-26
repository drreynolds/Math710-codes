function res = ARK_order_conditions(BE, BI, maxorder)
    % Usage: res = ARK_order_conditions(BE, BI, maxorder)
    %
    %        Inputs:
    %          BE is the explicit Butcher table, with components:
    %             BE.A -- the Butcher table matrix
    %             BE.b -- the solution coefficients
    %          BI is the implicit Butcher table, with the same components
    %          maxorder is optional, specifying the highest order of
    %             conditions to generate (1 <= maxorder <= 5, default 4)
    %
    %        Outputs:
    %          res is a cell array indexed by order q = 1,...,maxorder, where
    %             res{q} is a containers.Map mapping a label for each order-q
    %             condition to its residual (zero when the condition holds)
    %
    %        The abscissae for each table are computed as c = A*1 (internal
    %        consistency), so that the tables need not share their abscissae.
    %        Condition labels list the table used for each factor, e.g.,
    %        'bE.AI.cE' denotes (bE)^T AI cE - 1/6, and 'bI.cE*cI' denotes
    %        (bI)^T diag(cE) cI - 1/3; each '*' applies to everything on its
    %        right, so parentheses mark the one order-5 product of two matrix-
    %        vector terms, e.g., 'bE.(AE.cI)*(AI.cE)' denotes
    %        (bE)^T diag(AE cI) AI cE - 1/20.  Labels where all factors use the same
    %        table are the order conditions for that table alone; all others
    %        are coupling conditions.  Inputs may be double arrays, or sym
    %        arrays with exact (rational or symbolic) entries.
    %
% Function to check the order conditions (including the coupling conditions)
% for two-component additive Runge--Kutta (ARK) methods.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
if nargin < 3 || isempty(maxorder)
    maxorder = 4;
end

% set up tables and abscissae
A.E = sym(BE.A);  A.I = sym(BI.A);
b.E = sym(BE.b(:));  b.I = sym(BI.b(:));
s = size(A.E, 1);
if ((size(A.I, 1) ~= s) || (size(b.E, 1) ~= s) || (size(b.I, 1) ~= s))
    error('ARK_order_conditions ERROR: incompatible Butcher tables supplied');
end
one = sym(ones(s, 1));
c.E = A.E*one;  c.I = A.I*one;
R = @(p, q) sym(p)/sym(q);

% generate conditions
res = cell(1, maxorder);
for q = 1:maxorder
    res{q} = containers.Map('KeyType', 'char', 'ValueType', 'any');
end
for u = 'EI'
    res{1}(sprintf('b%s.1', u)) = dot(b.(u), one) - 1;
end
if (maxorder >= 2)
    for u = 'EI', for x = 'EI'
        res{2}(sprintf('b%s.c%s', u, x)) = dot(b.(u), c.(x)) - R(1,2);
    end, end
end
if (maxorder >= 3)
    for u = 'EI', for x = 'EI', for y = 'EI'
        if (x <= y)
            res{3}(sprintf('b%s.c%s*c%s', u, x, y)) = dot(b.(u), hadamard(c.(x), c.(y))) - R(1,3);
        end
        res{3}(sprintf('b%s.A%s.c%s', u, x, y)) = dot(b.(u), A.(x)*c.(y)) - R(1,6);
    end, end, end
end
if (maxorder >= 4)
    for u = 'EI', for x = 'EI', for y = 'EI', for z = 'EI'
        if (x <= y && y <= z)
            res{4}(sprintf('b%s.c%s*c%s*c%s', u, x, y, z)) = dot(b.(u), hadamard(c.(x), hadamard(c.(y), c.(z)))) - R(1,4);
        end
        res{4}(sprintf('b%s.c%s*A%s.c%s', u, x, y, z)) = dot(b.(u), hadamard(c.(x), A.(y)*c.(z))) - R(1,8);
        if (y <= z)
            res{4}(sprintf('b%s.A%s.c%s*c%s', u, x, y, z)) = dot(b.(u), A.(x)*hadamard(c.(y), c.(z))) - R(1,12);
        end
        res{4}(sprintf('b%s.A%s.A%s.c%s', u, x, y, z)) = dot(b.(u), A.(x)*A.(y)*c.(z)) - R(1,24);
    end, end, end, end
end
if (maxorder >= 5)
    for u = 'EI', for x = 'EI', for y = 'EI', for z = 'EI', for w = 'EI'
        if (x <= y && y <= z && z <= w)
            res{5}(sprintf('b%s.c%s*c%s*c%s*c%s', u, x, y, z, w)) = dot(b.(u), hadamard(c.(x), hadamard(c.(y), hadamard(c.(z), c.(w))))) - R(1,5);
        end
        if (x <= y)
            res{5}(sprintf('b%s.c%s*c%s*A%s.c%s', u, x, y, z, w)) = dot(b.(u), hadamard(c.(x), hadamard(c.(y), A.(z)*c.(w)))) - R(1,10);
        end
        if (z <= w)
            res{5}(sprintf('b%s.c%s*A%s.c%s*c%s', u, x, y, z, w)) = dot(b.(u), hadamard(c.(x), A.(y)*hadamard(c.(z), c.(w)))) - R(1,15);
        end
        res{5}(sprintf('b%s.c%s*A%s.A%s.c%s', u, x, y, z, w)) = dot(b.(u), hadamard(c.(x), A.(y)*A.(z)*c.(w))) - R(1,30);
        if ((x < z) || (x == z && y <= w))
            res{5}(sprintf('b%s.(A%s.c%s)*(A%s.c%s)', u, x, y, z, w)) = dot(b.(u), hadamard(A.(x)*c.(y), A.(z)*c.(w))) - R(1,20);
        end
        if (y <= z && z <= w)
            res{5}(sprintf('b%s.A%s.c%s*c%s*c%s', u, x, y, z, w)) = dot(b.(u), A.(x)*hadamard(c.(y), hadamard(c.(z), c.(w)))) - R(1,20);
        end
        res{5}(sprintf('b%s.A%s.c%s*A%s.c%s', u, x, y, z, w)) = dot(b.(u), A.(x)*hadamard(c.(y), A.(z)*c.(w))) - R(1,40);
        if (z <= w)
            res{5}(sprintf('b%s.A%s.A%s.c%s*c%s', u, x, y, z, w)) = dot(b.(u), A.(x)*A.(y)*hadamard(c.(z), c.(w))) - R(1,60);
        end
        res{5}(sprintf('b%s.A%s.A%s.A%s.c%s', u, x, y, z, w)) = dot(b.(u), A.(x)*A.(y)*A.(z)*c.(w)) - R(1,120);
    end, end, end, end, end
end

% simplify residuals
for q = 1:maxorder
    labels = keys(res{q});
    for k = 1:numel(labels)
        res{q}(labels{k}) = simplify(res{q}(labels{k}));
    end
end

    function d = dot(u, v)
        d = sym(0);
        for i = 1:s
            d = d + u(i)*v(i);
        end
    end

    function h = hadamard(u, v)
        h = sym(zeros(s, 1));
        for i = 1:s
            h(i) = u(i)*v(i);
        end
    end
end
