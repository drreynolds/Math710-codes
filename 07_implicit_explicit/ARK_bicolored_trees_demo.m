% Demo that uses ARK_conditions to count the coupling conditions for
% two-component additive Runge--Kutta (ARK) methods at each order, and
% ARK_condensed_conditions to print the condensed conditions themselves.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear

% highest orders for the count table and for printing the conditions
maxorder_count = 6;
maxorder_print = 6;

cases = {'general', false, false; 'bE=bI', true, false;
         'cE=cI', false, true; 'bE=bI, cE=cI', true, true};

% count the coupling conditions at each order, under each set of assumptions
fprintf('\nNumber of coupling conditions at each order:\n');
fprintf('  order');
fprintf('%15s', cases{:, 1});
fprintf('\n');
for q = 1:maxorder_count
    counts = zeros(1, size(cases, 1));
    for i = 1:size(cases, 1)
        conds = ARK_conditions(q, cases{i, 2}, cases{i, 3});
        counts(i) = sum([conds.coupling]);
    end
    fprintf('  %5i', q);
    fprintf('%15i', counts);
    fprintf('\n');
end

% print the condensed conditions themselves
for i = 1:size(cases, 1)
    fprintf('\nOrder conditions (%s), for all colors in {E,I}:\n', cases{i, 1});
    for q = 1:maxorder_print
        fprintf('  order %i:\n', q);
        conds = ARK_condensed_conditions(q, cases{i, 2}, cases{i, 3});
        for k = 1:numel(conds)
            fprintf('    %s = %s\n', conds(k).label, char(conds(k).rhs));
        end
    end
end
