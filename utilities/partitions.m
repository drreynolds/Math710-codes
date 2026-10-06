function parts = partitions(n, maxpart)
    % Usage: parts = partitions(n, maxpart)
    %
    %        Returns a cell array of the partitions of the integer n into parts
    %        no larger than maxpart (default n), as non-increasing row vectors.
    %
% Daniel R. Reynolds
% Math & Stat @ UMBC
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
