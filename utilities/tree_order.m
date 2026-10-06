function n = tree_order(t)
    % Usage: n = tree_order(t)
    %
    %        Returns the number of vertices in the tree t.
    %
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
n = 1;
for k = 1:numel(t)
    n = n + tree_order(t{k});
end
end
