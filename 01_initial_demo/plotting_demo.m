% MATLAB Plotting Introduction Script
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
clear

% get problem size from the command line, otherwise set to 201
N = str2double(input('Enter the vector size N >= 2 [default 201]: ', 's'));
if ~isfinite(N) || ~isreal(N) || N ~= round(N) || N < 2
    fprintf('Invalid or missing N, using the default value 201\n');
    N = 201;
end
fprintf('Running plotting demo using vectors of size N = %d\n', N);

% create x data
x = linspace(-1.0, 1.0, N).';

% create function data (first 5 odd-degree Chebyshev polynomials)
T = zeros(N, 5);
for j = 1:5
    T(:,j) = cos((2*j-1) * acos(x));
end

% create plots for visual diagnostics
figure(1);
plot(x, T);
xlabel('x');
ylabel('y');
title('Chebyshev polynomials');
legend('T_1(x)', 'T_3(x)', 'T_5(x)', 'T_7(x)', 'T_9(x)', 'Location', 'best');
saveas(gcf, 'figure1.png');

% manual colors and line styles
figure(2);
plot(x, T(:,1), 'b-',  'DisplayName', 'T_1(x)');
hold on;
plot(x, T(:,2), 'r--', 'DisplayName', 'T_3(x)');
plot(x, T(:,3), 'm:',  'DisplayName', 'T_5(x)');
plot(x, T(:,4), 'g-.', 'DisplayName', 'T_7(x)');
plot(x, T(:,5), 'c-',  'DisplayName', 'T_9(x)');
hold off;
xlabel('x');
ylabel('y');
title('Chebyshev polynomials');
legend('Location', 'best');
saveas(gcf, 'figure2.pdf');
