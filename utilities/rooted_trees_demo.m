% Script that calls rooted_trees to list the rooted trees of orders 1 through
% 5 in bracket notation, together with their densities.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC

clear

for q = 1:5
    trees = rooted_trees(q);
    fprintf('\nRooted trees of order %i (count = %i):\n', q, numel(trees));
    for k = 1:numel(trees)
        fprintf('  %-16s gamma = %i\n', tree_string(trees{k}), tree_density(trees{k}));
    end
end
