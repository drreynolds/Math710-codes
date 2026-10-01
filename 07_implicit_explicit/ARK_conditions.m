function conds = ARK_conditions(order, sameb, samec)
    % Usage: conds = ARK_conditions(order, sameb, samec)
    %
    %        Inputs:
    %          order is the order of the conditions to generate
    %          sameb is optional, indicating that bE = bI (default false)
    %          samec is optional, indicating that cE = cI (default false)
    %
    %        Outputs:
    %          conds is a struct array with fields label, rhs and coupling,
    %             one for each distinct order condition of the given order,
    %             where the condition is label = rhs, and coupling is true for
    %             coupling conditions and false for the order conditions of a
    %             single table
    %
% Function to generate the order conditions (including the coupling
% conditions) for two-component additive Runge--Kutta (ARK) methods, of any
% order, by enumerating bicolored rooted trees.  Optionally applies the
% simplifying assumptions bE = bI and/or cE = cI, under which many of the
% conditions coincide.
%
% Each node of a tree carries a color, E or I.  The root contributes the
% solution weights b of its color, each other node contributes the matrix A of
% its color, and each leaf therefore contributes the abscissae c = A*1 of its
% color.  If bE = bI, the color of the root is irrelevant; if cE = cI, the
% colors of the (non-root) leaves are irrelevant.
%
% In condition labels, '.' denotes a matrix product, e.g., 'bE.AI.cE' denotes
% (bE)^T AI cE.  A componentwise product with a vector is written as a product
% with the corresponding diagonal matrix, C = diag(c), e.g., 'bI.CE.cI' denotes
% (bI)^T diag(cE) cI, and 'bE.diag(AE.cI).AI.cE' denotes (bE)^T diag(AE cI) AI cE.
% A label without an E or I denotes a factor whose color is irrelevant under
% the simplifying assumptions.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
if nargin < 2 || isempty(sameb)
    sameb = false;
end
if nargin < 3 || isempty(samec)
    samec = false;
end

conds = struct('label', {}, 'rhs', {}, 'coupling', {});
trees = bicolored_trees(order, true, sameb, samec);
for k = 1:numel(trees)
    t = trees{k};
    coupling = (numel(tree_colors(t)) > 1);
    conds(end+1) = struct('label', tree_label(t), 'rhs', sym(1)/tree_density(t), 'coupling', coupling);
end
[~, idx] = sort({conds.label});
conds = conds(idx);
[~, idx] = sort(arrayfun(@(cond) double(cond.rhs), conds));
conds = conds(idx);
end


function parts = partitions(n, maxpart)
    % Usage: parts = partitions(n, maxpart)
    %
    %        Returns a cell array of the partitions of the integer n into parts
    %        no larger than maxpart (default n), as non-increasing row vectors.
    %
if nargin < 2 || isempty(maxpart)
    maxpart = n;
end
if (n == 0)
    parts = {[]};
    return
end
parts = {};
for k = min(n, maxpart):-1:1
    rest = partitions(n-k, k);
    for j = 1:numel(rest)
        parts{end+1} = [k, rest{j}];
    end
end
end


function combos = combinations_with_replacement(items, m)
    % Usage: combos = combinations_with_replacement(items, m)
    %
    %        Returns a cell array of every multiset of m entries from the cell
    %        array items, each as a 1-by-m cell array.
    %
if (m == 0)
    combos = {{}};
    return
end
combos = {};
for i = 1:numel(items)
    rest = combinations_with_replacement(items(i:end), m-1);
    for j = 1:numel(rest)
        combos{end+1} = [items(i), rest{j}];
    end
end
end


function trees = bicolored_trees(n, root, sameb, samec)
    % Usage: trees = bicolored_trees(n, root, sameb, samec)
    %
    %        Inputs:
    %          n is the number of nodes in each tree
    %          root indicates whether these trees are the full trees (true),
    %             or subtrees below the root (false)
    %          sameb indicates that bE = bI
    %          samec indicates that cE = cI
    %
    %        Outputs:
    %          trees is a sorted cell array of the distinct bicolored rooted
    %             trees with n nodes, each stored as a character string
    %             [color '(' children ')'], where color is 'E', 'I', or '-'
    %             (when the color is irrelevant), and children is the
    %             concatenation of the sorted strings for its subtrees
    %
persistent cache
if isempty(cache)
    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
end
key = sprintf('%i,%i,%i,%i', n, root, sameb, samec);
if isKey(cache, key)
    trees = cache(key);
    return
end

trees = {};
parts = partitions(n-1);
for p = 1:numel(parts)
    part = parts{p};
    % count the number of children of each size
    sizes = reshape(unique(part), 1, []);
    % form every multiset of subtrees having these sizes
    childlists = {{}};
    for k = sizes
        m = sum(part == k);
        subtrees = bicolored_trees(k, false, sameb, samec);
        combos = combinations_with_replacement(subtrees, m);
        newlists = {};
        for a = 1:numel(childlists)
            for b = 1:numel(combos)
                newlists{end+1} = [childlists{a}, combos{b}];
            end
        end
        childlists = newlists;
    end
    % color the root of each resulting tree
    for j = 1:numel(childlists)
        children = sort(childlists{j});
        leaf = isempty(children);
        if ((root && sameb) || ((~root) && leaf && samec))
            colors = {''};
        else
            colors = {'E', 'I'};
        end
        for i = 1:numel(colors)
            trees{end+1} = tree_string(colors{i}, children);
        end
    end
end
trees = unique(trees);
cache(key) = trees;
end


function t = tree_string(color, children)
    % Usage: t = tree_string(color, children)
    %
    %        Returns the string for the tree with the given root color ('E',
    %        'I', or '' when irrelevant) and cell array of subtree strings.
    %
if isempty(color)
    color = '-';
end
t = [color, '(', children{:}, ')'];
end


function [color, children] = tree_parts(t)
    % Usage: [color, children] = tree_parts(t)
    %
    %        Returns the root color ('E', 'I', or '' when irrelevant) and the
    %        cell array of subtree strings for the tree string t.
    %
color = t(1);
if (color == '-')
    color = '';
end
children = {};
i = 3;
while (i < numel(t))
    start = i;
    depth = 0;
    i = i + 1;
    while true
        if (t(i) == '(')
            depth = depth + 1;
        elseif (t(i) == ')')
            depth = depth - 1;
        end
        if (depth == 0)
            break
        end
        i = i + 1;
    end
    children{end+1} = t(start:i);
    i = i + 1;
end
end


function colors = tree_colors(t)
    % Usage: colors = tree_colors(t)
    %
    %        Returns the relevant colors ('E' and/or 'I') used in the tree t.
    %
[color, children] = tree_parts(t);
colors = color;
for k = 1:numel(children)
    colors = union(colors, tree_colors(children{k}));
end
end


function n = tree_order(t)
    % Usage: n = tree_order(t)
    %
    %        Returns the number of nodes in the tree t.
    %
[~, children] = tree_parts(t);
n = 1;
for k = 1:numel(children)
    n = n + tree_order(children{k});
end
end


function gamma = tree_density(t)
    % Usage: gamma = tree_density(t)
    %
    %        Returns the density gamma(t) of the tree t, so that the order
    %        condition for t is Phi(t) = 1/gamma(t).
    %
[~, children] = tree_parts(t);
gamma = tree_order(t);
for k = 1:numel(children)
    gamma = gamma * tree_density(children{k});
end
end


function label = vector_label(children)
    % Usage: label = vector_label(children)
    %
    %        Returns the label for the vector formed by the elementwise product
    %        of the terms contributed by each child subtree.  Leaves contribute
    %        c (listed first), and all other subtrees contribute A times the
    %        vector for their own children.  Every term but the last multiplies
    %        the terms to its right elementwise, and so is written as a
    %        diagonal matrix, C = diag(c) or diag(A...).
    %
leaves = {};
terms = {};
for k = 1:numel(children)
    [color, grandchildren] = tree_parts(children{k});
    if isempty(grandchildren)
        leaves{end+1} = color;
    else
        terms{end+1} = ['A', color, '.', vector_label(grandchildren)];
    end
end
factors = [strcat('C', leaves), strcat('diag(', terms, ')')];
if ~isempty(terms)
    factors{end} = terms{end};
else
    factors{end} = ['c', leaves{end}];
end
label = strjoin(factors, '.');
end


function label = tree_label(t)
    % Usage: label = tree_label(t)
    %
    %        Returns the label for the order condition corresponding to the
    %        full tree t.
    %
[color, children] = tree_parts(t);
if isempty(children)
    label = ['b', color, '.1'];
    return
end
label = ['b', color, '.', vector_label(children)];
end
