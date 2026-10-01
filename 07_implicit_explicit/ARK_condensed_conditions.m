function conds = ARK_condensed_conditions(order, sameb, samec)
    % Usage: conds = ARK_condensed_conditions(order, sameb, samec)
    %
    %        Inputs:
    %          order is the order of the conditions to generate
    %          sameb is optional, indicating that bE = bI (default false)
    %          samec is optional, indicating that cE = cI (default false)
    %
    %        Outputs:
    %          conds is a struct array with fields label and rhs, one for each
    %             rooted tree of the given order, where the condition
    %             label = rhs must hold for every choice of the placeholder
    %             colors in {E,I}
    %
% Function to generate the order conditions (including the coupling
% conditions) for two-component additive Runge--Kutta (ARK) methods in a
% condensed form, once per rooted tree, using placeholder colors (nu, mu,
% lambda, ...) for its vertices, where each condition must hold for every
% choice of these colors in {E,I}.  Optionally applies the simplifying
% assumptions bE = bI and/or cE = cI, under which the root and/or the leaves
% no longer need colors of their own.
%
% In condition labels, '.' denotes a matrix product, e.g., 'bν.Aμ.cλ' denotes
% (bν)^T Aμ cλ.  A componentwise product with a vector is written as a product
% with the corresponding diagonal matrix, C = diag(c), e.g., 'bν.Cμ.cλ' denotes
% (bν)^T diag(cμ) cλ, and 'bν.diag(Aμ.cλ).Aσ.cκ' denotes (bν)^T diag(Aμ cλ) Aσ cκ.
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

conds = struct('label', {}, 'rhs', {});
trees = GARK_rooted_trees(order);
for k = 1:numel(trees)
    t = trees{k};
    conds(end+1) = struct('label', condensed_label(t, sameb, samec), 'rhs', sym(1)/tree_density(t));
end
[~, idx] = sort({conds.label});
conds = conds(idx);
[~, idx] = sort(arrayfun(@(cond) double(cond.rhs), conds));
conds = conds(idx);
end


function n = tree_order(t)
    % Usage: n = tree_order(t)
    %
    %        Returns the number of vertices in the (uncolored) rooted tree t.
    %
n = 1;
for k = 1:numel(t)
    n = n + tree_order(t{k});
end
end


function gamma = tree_density(t)
    % Usage: gamma = tree_density(t)
    %
    %        Returns the density gamma(t) of the (uncolored) rooted tree t, so
    %        that the order condition for t is Phi(t) = 1/gamma(t).
    %
gamma = tree_order(t);
for k = 1:numel(t)
    gamma = gamma * tree_density(t{k});
end
end


function label = condensed_label(t, sameb, samec)
    % Usage: label = condensed_label(t, sameb, samec)
    %
    %        Inputs:
    %          t is an (uncolored) rooted tree, from GARK_rooted_trees
    %          sameb indicates that bE = bI
    %          samec indicates that cE = cI
    %
    %        Outputs:
    %          label is the left-hand side of the condensed order condition for
    %             t, with placeholder colors assigned to the vertices in the
    %             order in which they appear in the label, and omitted for the
    %             vertices whose colors are irrelevant
    %
% placeholder color names, in the order that they are assigned to vertices
placeholders = {'ν', 'μ', 'λ', 'σ', 'κ', 'ρ', 'τ', 'ω'};
count = 0;

if sameb
    x = '';
else
    x = new_color();
end
if isempty(t)
    label = ['b', x, '.1'];
    return
end
label = ['b', x, '.', vector(t)];

    function name = new_color()
        count = count + 1;
        name = placeholders{count};
    end

    function v = vector(children)
        % leaves contribute c (listed first), and all other subtrees contribute
        % A times the vector for their own children
        leaves = {};
        for k = 1:numel(children)
            if isempty(children{k})
                if samec
                    leaves{end+1} = '';
                else
                    leaves{end+1} = new_color();
                end
            end
        end
        terms = {};
        for k = 1:numel(children)
            if ~isempty(children{k})
                y = new_color();
                terms{end+1} = ['A', y, '.', vector(children{k})];
            end
        end
        % every factor but the last is written as a diagonal matrix
        factors = [strcat('C', leaves), strcat('diag(', terms, ')')];
        if ~isempty(terms)
            factors{end} = terms{end};
        else
            factors{end} = ['c', leaves{end}];
        end
        v = strjoin(factors, '.');
    end
end
