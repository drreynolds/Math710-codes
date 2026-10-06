function gamma = tree_density(t)
    % Usage: gamma = tree_density(t)
    %
    %        Returns the density gamma(t) of the tree t, so that the order
    %        condition for t is Phi(t) = 1/gamma(t).
    %
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
gamma = tree_order(t);
for k = 1:numel(t)
    gamma = gamma * tree_density(t{k});
end
end
