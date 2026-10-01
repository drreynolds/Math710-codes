% Demo that uses GARK_order_conditions, GARK_order, and FSRK_tableau to
% examine an ARK method written in GARK form, and several fractional-step
% methods from the lecture notes.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
R = @(p, q) sym(p)/sym(q);

% ARS (2,2,2), written in GARK form with A^{sigma,nu} = A^{nu}
g = (2 - sqrt(sym(2)))/2;
d = 1 - 1/(2*g);
AE = [0, 0, 0; g, 0, 0; d, 1-d, 0];
AI = [0, 0, 0; 0, g, 0; 0, 1-g, g];
ARS222 = {{AE, AI; AE, AI}, {[d; 1-d; 0], [0; 1-g; g]}};

% sub-integrators for the fractional-step methods
FE = struct('A', sym(0), 'b', sym(1));
Heun = struct('A', [0, 0; 1, 0], 'b', [R(1,2); R(1,2)]);
RK4 = struct('A', [0, 0, 0, 0; R(1,2), 0, 0, 0; 0, R(1,2), 0, 0; 0, 0, 1, 0], ...
             'b', [R(1,6); R(1,3); R(1,3); R(1,6)]);

% fractional-step coefficients from the lecture notes
LT = {sym(1), sym(1)};
Strang = {[R(1,2), R(1,2)], sym([1, 0])};
Ruth = {[R(7,24), R(3,4), -R(1,24)], [R(2,3), -R(2,3), 1]};

tests = {'ARS(2,2,2) in GARK form', ARS222;
         'Lie-Trotter + forward Euler', fsrk(LT, {FE, FE});
         'Lie-Trotter + RK4', fsrk(LT, {RK4, RK4});
         'Strang + RK4', fsrk(Strang, {RK4, RK4});
         'Ruth + RK4', fsrk(Ruth, {RK4, RK4});
         'Ruth + Heun', fsrk(Ruth, {Heun, Heun})};
for i = 1:size(tests, 1)
    name = tests{i, 1};
    A = tests{i, 2}{1};
    b = tests{i, 2}{2};
    fprintf('\n%s:\n', name);
    M = numel(b);
    pbase = zeros(1, M);
    for j = 1:M
        pbase(j) = GARK_order({A{j,j}}, {b{j}}, 1e-8);
    end
    [p, failed] = GARK_order(A, b, 1e-8);
    fprintf('  base method orders = %s,  GARK order = %i\n', strjoin(arrayfun(@num2str, pbase, 'UniformOutput', false), ', '), p);
    if (p < min(pbase))
        fprintf('  failing coupling conditions: %s\n', strjoin(failed, ', '));
    end
end


function tab = fsrk(alpha, B)
    % Usage: tab = fsrk(alpha, B)
    %
    %        Returns the GARK tableau of a fractional-step method, as the cell
    %        array {A, b}, using FSRK_tableau.
    %
[A, b] = FSRK_tableau(alpha, B);
tab = {A, b};
end
