function conds = GARK_conditions(order, internal, latex)
    % Usage: conds = GARK_conditions(order, internal, latex)
    %
    %        Inputs:
    %          order is the order of the conditions to generate
    %          internal is optional, indicating that the method is internally
    %             consistent, c^{sigma,nu} = c^{sigma} (default false)
    %          latex is optional, generating the labels in LaTeX (default false)
    %
    %        Outputs:
    %          conds is a struct array with fields label and rhs, one for each
    %             rooted tree of the given order, where the condition
    %             label = rhs must hold for every choice of the placeholder
    %             colors in {1,...,M}
    %
% Function to generate the order conditions for generalized-structure
% additive Runge--Kutta (GARK) methods with an arbitrary number M of
% partitions, of any order, by enumerating rooted trees.  Each condition is
% written once per rooted tree, using placeholder colors (sigma, nu, mu, ...)
% for its vertices, and must hold for every choice of these colors in
% {1,...,M}.  Optionally applies internal consistency, c^{sigma,nu} =
% c^{sigma}, under which the leaves no longer need colors of their own.
%
% Each vertex of a tree carries a color.  The root, of color sigma,
% contributes the solution weights b^{sigma}; a vertex of color nu attached to
% a parent of color mu contributes A^{mu,nu}; and so a leaf of color nu
% attached to a parent of color mu contributes c^{mu,nu} = A^{mu,nu}*1.
%
% In condition labels, '.' denotes a matrix product, e.g., 'b{σ}.A{σ,ν}.c{ν,μ}'
% denotes (b^{σ})^T A^{σ,ν} c^{ν,μ}.  A componentwise product with a vector is
% written as a product with the corresponding diagonal matrix, C = diag(c),
% e.g., 'b{σ}.C{σ,ν}.c{σ,μ}' denotes (b^{σ})^T diag(c^{σ,ν}) c^{σ,μ}, and
% 'b{σ}.diag(A{σ,ν}.c{ν,μ}).A{σ,λ}.c{λ,κ}' denotes
% (b^{σ})^T diag(A^{σ,ν} c^{ν,μ}) A^{σ,λ} c^{λ,κ}.  Labels may instead be
% generated in LaTeX, using the macros of the lecture notes.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
if nargin < 2 || isempty(internal)
    internal = false;
end
if nargin < 3 || isempty(latex)
    latex = false;
end

% rooted_trees and tree_density are in the utilities folder
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'utilities'));

conds = struct('label', {}, 'rhs', {});
trees = rooted_trees(order);
for k = 1:numel(trees)
    t = trees{k};
    conds(end+1) = struct('label', condition_label(t, internal, latex), 'rhs', sym(1)/tree_density(t));
end
[~, idx] = sort({conds.label});
conds = conds(idx);
[~, idx] = sort(arrayfun(@(cond) double(cond.rhs), conds));
conds = conds(idx);
end


function label = condition_label(t, internal, latex)
    % Usage: label = condition_label(t, internal, latex)
    %
    %        Inputs:
    %          t is a rooted tree
    %          internal indicates that the method is internally consistent,
    %             c^{sigma,nu} = c^{sigma}
    %          latex indicates that the label should be generated in LaTeX
    %
    %        Outputs:
    %          label is the left-hand side of the order condition for t, with
    %             placeholder colors assigned to the vertices in the order in
    %             which they appear in the label
    %
% placeholder color names, in the order that they are assigned to vertices
text_colors = {'σ', 'ν', 'μ', 'λ', 'κ', 'ρ', 'τ', 'ω'};
latex_colors = {'\sigma', '\nu', '\mu', '\lambda', '\kappa', '\rho', '\tau', '\omega'};
if latex
    names = latex_colors;
else
    names = text_colors;
end
count = 0;

x = new_color();
if latex
    if isempty(t)
        label = ['(\bvec', sup({x}), ')^T\onevec'];
        return
    end
    label = ['(\bvec', sup({x}), ')^T', vector(t, x)];
    return
end
if isempty(t)
    label = ['b', sup({x}), '.1'];
    return
end
label = ['b', sup({x}), '.', vector(t, x)];

    function name = new_color()
        count = count + 1;
        name = names{count};
    end

    function s = sup(colors)
        if latex
            s = ['^{\{', strjoin(colors, ','), '\}}'];
        else
            s = ['{', strjoin(colors, ','), '}'];
        end
    end

    function v = vector(children, x)
        % leaves contribute c (listed first), and all other subtrees contribute
        % A times the vector for their own children
        kinds = {};
        sups = {};
        vecs = {};
        for k = 1:numel(children)
            if isempty(children{k})
                if internal
                    colors = {x};
                else
                    colors = {x, new_color()};
                end
                kinds{end+1} = 'leaf';
                sups{end+1} = sup(colors);
                vecs{end+1} = '';
            end
        end
        for k = 1:numel(children)
            if ~isempty(children{k})
                y = new_color();
                kinds{end+1} = 'term';
                sups{end+1} = sup({x, y});
                vecs{end+1} = vector(children{k}, y);
            end
        end
        if latex
            v = '';
            for k = 1:numel(kinds)
                last = (k == numel(kinds));
                if strcmp(kinds{k}, 'leaf')
                    if last
                        v = [v, '\cvec', sups{k}];
                    else
                        v = [v, 'C', sups{k}];
                    end
                elseif last
                    v = [v, 'A', sups{k}, vecs{k}];
                else
                    v = [v, '\operatorname{diag}\(A', sups{k}, vecs{k}, '\)'];
                end
            end
            return
        end
        labels = {};
        for k = 1:numel(kinds)
            last = (k == numel(kinds));
            if strcmp(kinds{k}, 'leaf')
                if last
                    labels{end+1} = ['c', sups{k}];
                else
                    labels{end+1} = ['C', sups{k}];
                end
            elseif last
                labels{end+1} = ['A', sups{k}, '.', vecs{k}];
            else
                labels{end+1} = ['diag(A', sups{k}, '.', vecs{k}, ')'];
            end
        end
        v = strjoin(labels, '.');
    end
end
