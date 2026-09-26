% Script to solve the IVP system
%    u1'' = u1 + 2*u2' - muh*(u1+mu)/D1 - mu*(u1-muh)/D2,
%    u2'' = u2 - 2*u1' - muh*u2/D1 - mu*u2/D2,
% with initial conditions
%    u1(0) = 0.994,  u2(0) = 0,  u1'(0) = 0,
%    u2'(0) = -2.00158510637908252240537862224
% over the time interval [0,17.1]
%
% Let y0 = u1,  y1 = u1',  y2 = u2,  y3 = y2', then this is equivalent to
%    y0' = y1,
%    y1' = y0 + 2*y3 - muh*(y0+mu)/D1 - mu*(y0-muh)/D2,
%    y2' = y3,
%    y3' = y2 - 2*y1 - muh*y2/D1 - mu*y2/D2,
% with initial conditions
%    y0(0) = 0.994,  y1(0) = 0,  y2(0) = 0,
%    y3(0) = -2.00158510637908252240537862224
% over the time interval [0,17.1]
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear
t0 = 0.0;
tf = 17.1;
y0 = [0.994; 0.0; 0.0; -2.00158510637908252240537862224];

Nout = 101;   % includes initial condition
% set output times for the experiment
tspan = linspace(t0, tf, Nout).';

rtol = 1e-6;
atol = 1e-12;

% Compare two embedded adaptive ERK methods against several fixed-step ERK4
% runs on the same output grid.
BS = AdaptERK(@f, y0, AdaptERK.BogackiShampine(), rtol, atol, [], [], [], [], [], true);
DP = AdaptERK(@f, y0, AdaptERK.DormandPrince(), rtol, atol, [], [], [], [], [], true);
E4 = ERK(@f, ERK.ERK4());

fprintf('\nAdaptive Bogacki-Shampine solver:\n');
[Y_BS, success] = BS.Evolve(tspan, y0);
if ~success
    fprintf('  solve failed\n');
end
step_hist_BS = BS.get_step_history();
fprintf('  steps = %5d  fails = %2d\n\n', BS.get_num_steps(), BS.get_num_error_failures());

fprintf('\nAdaptive Dormand-Prince solver:\n');
[Y_DP, success] = DP.Evolve(tspan, y0);
if ~success
    fprintf('  solve failed\n');
end
step_hist_DP = DP.get_step_history();
fprintf('  steps = %5d  fails = %2d\n\n', DP.get_num_steps(), DP.get_num_error_failures());

fprintf('\nERK4 runs:\n');
fprintf('  100 steps:\n');
[Y_erk4_100, success] = E4.Evolve(tspan, y0, tf/100);
if ~success, fprintf('    solve failed\n'); end

fprintf('  1000 steps:\n');
[Y_erk4_1000, success] = E4.Evolve(tspan, y0, tf/1000);
if ~success, fprintf('    solve failed\n'); end

fprintf('  10000 steps:\n');
[Y_erk4_10000, success] = E4.Evolve(tspan, y0, tf/10000);
if ~success, fprintf('    solve failed\n'); end

fprintf('  20000 steps:\n');
[Y_erk4_20000, success] = E4.Evolve(tspan, y0, tf/20000);
if ~success, fprintf('    solve failed\n'); end

% Plot adaptive orbits and step histories, then compare with fixed-step ERK4.
figure();
plot(Y_BS(:,1), Y_BS(:,3));
xlabel('u_1');
ylabel('u_2');
title(sprintf('Orbit (Bogacki-Shampine, %d steps)', BS.get_num_steps()));
saveas(gcf, 'adaptive_BS_orbit.png');

figure();
plot(Y_DP(:,1), Y_DP(:,3));
xlabel('u_1');
ylabel('u_2');
title(sprintf('Orbit (Dormand-Prince %d steps)', DP.get_num_steps()));
saveas(gcf, 'adaptive_DP_orbit.png');

figure();
plot(step_hist_BS.t, step_hist_BS.h, 'b-', 'DisplayName', 'Bogacki-Shampine');
hold on;
plot(step_hist_DP.t, step_hist_DP.h, 'r-', 'DisplayName', 'Dormand-Prince');
plotFailures(step_hist_BS, 'bx');
plotFailures(step_hist_DP, 'rx');
hold off;
xlabel('t');
ylabel('h');
title('Adaptive step history');
legend('Location', 'best');
saveas(gcf, 'adaptive_ERK_steps.png');

figure();
plot(Y_erk4_100(:,1), Y_erk4_100(:,3));
xlabel('u_1');
ylabel('u_2');
title('Orbit (ERK4, 100 steps)');
saveas(gcf, 'orbit_100.png');

figure();
plot(Y_erk4_1000(:,1), Y_erk4_1000(:,3));
xlabel('u_1');
ylabel('u_2');
title('Orbit (ERK4, 1000 steps)');
saveas(gcf, 'orbit_1000.png');

figure();
plot(Y_erk4_10000(:,1), Y_erk4_10000(:,3));
xlabel('u_1');
ylabel('u_2');
title('Orbit (ERK4, 10000 steps)');
saveas(gcf, 'orbit_10000.png');

figure();
plot(Y_erk4_20000(:,1), Y_erk4_20000(:,3));
xlabel('u_1');
ylabel('u_2');
title('Orbit (ERK4, 20000 steps)');
saveas(gcf, 'orbit_20000.png');

function val = f(~, y)
    % ODE RHS function

    mu = 0.012277471;
    muh = 1.0 - mu;
    D1 = ((y(1)+mu)^2 + y(3)^2)^1.5;
    D2 = ((y(1)-muh)^2 + y(3)^2)^1.5;
    val = [y(2); ...
           y(1) + 2.0*y(4) - muh*(y(1)+mu)/D1 - mu*(y(1)-muh)/D2; ...
           y(4); ...
           y(3) - 2.0*y(2) - muh*y(3)/D1 - mu*y(3)/D2];
end

function plotFailures(step_hist, marker)
    idx = step_hist.err > 1.0;
    if any(idx)
        plot(step_hist.t(idx), step_hist.h(idx), marker, 'HandleVisibility', 'off');
    end
end
