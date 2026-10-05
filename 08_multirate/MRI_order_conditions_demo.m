% Demo that uses MRI_order to check the order conditions of the explicit
% MRI-GARK methods in MRI.m, through their GARK representation.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear

maxorder = 5;
tests = {'MRI-GARK-ERK22a', MRI.MRIGARKERK22a();
         'MRI-GARK-ERK33a', MRI.MRIGARKERK33a();
         'MRI-GARK-ERK45a', MRI.MRIGARKERK45a()};
for i = 1:size(tests, 1)
    name = tests{i, 1};
    C = tests{i, 2};
    fprintf('\n%s:\n', name);
    [pbase, p, failed] = MRI_order(C, maxorder, 1e-8);
    fprintf('  slow base method order = %i,  MRI-GARK order = %i\n', pbase, p);
    if (p < min(pbase, maxorder))
        fprintf('  failing coupling conditions: %s\n', strjoin(failed, ', '));
    end
end
