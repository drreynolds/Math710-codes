function fd_maker_demo()
% MATLAB teaching demo for fd maker.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
fprintf('\nCentered 3-point stencil for the first derivative:\n');
stencil = [-1, 0, 1];
[coeffs, errorterm] = fd_maker(stencil, 1);
printStencil(stencil, coeffs, errorterm);

fprintf('\nOne-sided (right facing) 6-point stencil for the second derivative:\n');
stencil = [0, 1, 2, 3, 4, 5];
[coeffs, errorterm] = fd_maker(stencil, 2);
printStencil(stencil, coeffs, errorterm);

fprintf('\nLopsided 8-point stencil for the third derivative:\n');
stencil = [-2, -1, 0, 1, 2, 3, 4, 5];
[coeffs, errorterm] = fd_maker(stencil, 3);
printStencil(stencil, coeffs, errorterm);
end

function printStencil(stencil, coeffs, errorterm)
    fprintf('  stencil = ');
    fprintf('%g ', stencil);
    fprintf('\n  coefficients = ');
    fprintf('%.16g ', coeffs);
    fprintf('\n  error term = %.16g*h^%d\n', errorterm.coefficient, errorterm.power);
end
