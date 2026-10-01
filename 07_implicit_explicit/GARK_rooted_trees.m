function trees = GARK_rooted_trees(n)
    % Usage: trees = GARK_rooted_trees(n)
    %
    %        Returns a sorted cell array of the distinct (uncolored) rooted
    %        trees with n vertices, each stored as the sorted 1-by-k cell array
    %        of its subtrees (so that a single vertex is stored as {}).
    %
% Function to enumerate the rooted trees used by GARK_conditions and
% GARK_colored_trees_demo.  Trees are sorted using a string key in which each
% vertex is written as '1', followed by the keys of its subtrees, followed by
% '0'; this orders the trees (and the subtrees of each vertex) exactly as the
% sorted tuples of GARK_colored_trees.py.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
persistent cache
if isempty(cache)
    cache = containers.Map('KeyType', 'double', 'ValueType', 'any');
end
if isKey(cache, n)
    trees = cache(n);
    return
end

trees = {};
keys = {};
parts = partitions(n-1);
for p = 1:numel(parts)
    part = parts{p};
    % count the number of children of each size
    sizes = reshape(unique(part), 1, []);
    % form every multiset of subtrees having these sizes
    childlists = {{}};
    for k = sizes
        m = sum(part == k);
        combos = combinations_with_replacement(GARK_rooted_trees(k), m);
        newlists = {};
        for a = 1:numel(childlists)
            for b = 1:numel(combos)
                newlists{end+1} = [childlists{a}, combos{b}];
            end
        end
        childlists = newlists;
    end
    for j = 1:numel(childlists)
        children = childlists{j};
        [~, idx] = sort(cellfun(@tree_key, children, 'UniformOutput', false));
        t = children(idx);
        trees{end+1} = t;
        keys{end+1} = tree_key(t);
    end
end
[~, idx] = unique(keys);
trees = trees(idx);
cache(n) = trees;
end


function key = tree_key(t)
    % Usage: key = tree_key(t)
    %
    %        Returns the sorting key for the tree t.
    %
key = '1';
for k = 1:numel(t)
    key = [key, tree_key(t{k})];
end
key = [key, '0'];
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
