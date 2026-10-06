function s = tree_string(t)
    % Usage: s = tree_string(t)
    %
    %        Returns the tree t in bracket notation, e.g., '[•,[•]]', where '•'
    %        denotes a single vertex.
    %
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
if isempty(t)
    s = '•';
    return
end
s = ['[', strjoin(cellfun(@tree_string, t, 'UniformOutput', false), ','), ']'];
end
