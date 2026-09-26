% Demo that uses ARK_order_conditions, ARK_order, and ARK_stiff_limit to
% examine a few ARK methods from the lecture notes.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
R = @(p, q) sym(p)/sym(q);

% ARS (1,2,2): explicit and (padded) implicit midpoint
BE = struct('A', [0, 0; R(1,2), 0], 'b', [0; 1]);
BI = struct('A', [0, 0; 0, R(1,2)], 'b', [0; 1]);
ARS122 = {BE, BI};

% ARS (2,2,2)
g = (2 - sqrt(sym(2)))/2;
d = 1 - 1/(2*g);
BE = struct('A', [0, 0, 0; g, 0, 0; d, 1-d, 0], 'b', [d; 1-d; 0]);
BI = struct('A', [0, 0, 0; 0, g, 0; 0, 1-g, g], 'b', [0; 1-g; g]);
ARS222 = {BE, BI};

% ARS (3,4,3), using the 10-digit coefficients from Ascher et al. (1997)
g = 0.4358665215;
b1 = 1.208496649;
b2 = -0.644363171;
BE = struct('A', [0.0, 0.0, 0.0, 0.0;
                  g, 0.0, 0.0, 0.0;
                  0.3212788860, 0.3966543747, 0.0, 0.0;
                  -0.105858296, 0.5529291479, 0.5529291479, 0.0], ...
            'b', [0.0; b1; b2; g]);
BI = struct('A', [0.0, 0.0, 0.0, 0.0;
                  0.0, g, 0.0, 0.0;
                  0.0, 0.2820667392, g, 0.0;
                  0.0, b1, b2, g], ...
            'b', [0.0; b1; b2; g]);
ARS343 = {BE, BI};

% coupling-failure demonstration pairs from the lecture notes
Heun = struct('A', [0, 0; 1, 0], 'b', [R(1,2); R(1,2)]);
RK4 = struct('A', [0, 0, 0, 0; R(1,2), 0, 0, 0; 0, R(1,2), 0, 0; 0, 0, 1, 0], ...
             'b', [R(1,6); R(1,3); R(1,3); R(1,6)]);
ERK3 = struct('A', [0, 0, 0, 0; R(1,2), 0, 0, 0; 0, R(1,2), 0, 0; 1, 0, 0, 0], ...
              'b', [R(1,6); 0; R(2,3); R(1,6)]);
ESDIRK3 = struct('A', [0, 0, 0, 0; R(1,6), R(1,3), 0, 0; ...
                       R(1,2), -R(1,3), R(1,3), 0; -R(2,3), R(2,3), R(2,3), R(1,3)], ...
                 'b', [R(1,6); 0; R(2,3); R(1,6)]);

tests = {'ARS(1,2,2)', ARS122; 'ARS(2,2,2)', ARS222; 'ARS(3,4,3)', ARS343;
         'Heun + implicit midpoint', {Heun, ARS122{2}};
         'RK4 + ESDIRK3', {RK4, ESDIRK3}; 'ERK3 + ESDIRK3', {ERK3, ESDIRK3}};
for i = 1:size(tests, 1)
    name = tests{i, 1};
    BE = tests{i, 2}{1};
    BI = tests{i, 2}{2};
    fprintf('\n%s:\n', name);
    pE = ARK_order(BE, BE, 1e-8);
    pI = ARK_order(BI, BI, 1e-8);
    [p, failed] = ARK_order(BE, BI, 1e-8);
    fprintf('  explicit order = %i,  implicit order = %i,  ARK order = %i\n', pE, pI, p);
    if (p < min(pE, pI))
        fprintf('  failing coupling conditions: %s\n', strjoin(failed, ', '));
    end
    [Rf, Rlim] = ARK_stiff_limit(BE, BI);
    fprintf('  R(zE, zI -> -infinity) = %s\n', char(vpa(expand(Rlim), 6)));
end
