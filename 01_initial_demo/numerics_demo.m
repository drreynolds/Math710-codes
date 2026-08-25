function numpy_demo()
% Basic numpy usage demo script
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
% initial setup
%
 a = zeros(1, 5);
fprintf('writing array of zeros: ');
disp(a);

% create a vector with specific entries
b = [0.1, 0.2, 0.3, 0.4, 0.5];
fprintf('writing array of 0.1, 0.2, 0.3, 0.4, 0.5: ');
disp(b);

% create a vector using linspace
c = linspace(1.0, 5.0, 5);
fprintf('writing array of 1, 2, 3, 4, 5: ');
disp(c);

if numel(b) ~= 5
    error('incorrect matrix size');
end

% edit entries of a
for i = 1:numel(a)
    a(i) = 5.0 * i + 5.0;
end
fprintf('entries of a, one at a time (should give 10, 15, 20, 25, 30):\n');
for i = 1:numel(a)
    fprintf('  %.1f\n', a(i));
end

% save/load test
fprintf('writing/reading this same vector to/from the file a_data.txt\n');
writeArray(a.', 'a_data.txt');

tol = 2e-15;
read_test1a = rand(3,4);
writeArray(read_test1a, 'tmp.txt');
read_test1b = readArray('tmp.txt');
read_test1_error = read_test1a - read_test1b;
if norm(read_test1_error, inf) < tol
    fprintf('  save/load test 1 passed\n');
else
    fprintf('  save/load test 1 failed, ||error|| = %.3e\n', norm(read_test1_error, inf));
end

% copy semantics
B = a;
fprintf('B = a copy, should give 10, 15, 20, 25, 30: ');
disp(B);

a(5) = 31.0;
fprintf('updating the 5th entry of a to be 31:\n');
fprintf('   a = '); disp(a);
fprintf('B should not have changed:\n');
fprintf('   B = '); disp(B);
a(5) = 30.0;

% slice-style updates
B2 = a(2:4);
fprintf('B2 = a(2:4): ');
disp(B2);
B2(:) = [4.0, 3.0, 2.0];
a(2:4) = B2;
fprintf('span copy back into a, should have entries 10 4 3 2 30\n');
fprintf('   '); disp(a);
a(2:4) = [15.0, 20.0, 25.0];

% arithmetic operators
fprintf('Testing vector add, should give 1.1, 2.2, 3.3, 4.4, 5.5\n');
b = b + c;
fprintf('   '); disp(b);

fprintf('Testing scalar add, should give 2, 3, 4, 5, 6\n');
c = c + 1.0;
fprintf('   '); disp(c);

fprintf('Testing vector subtract, should be 8, 12, 16, 20, 24\n');
a = a - c;
fprintf('   '); disp(a);

fprintf('Testing scalar subtract, should be 0, 1, 2, 3, 4\n');
c = c - 2.0;
fprintf('   '); disp(c);

fprintf('Testing scalar multiply, should be 0, 5, 10, 15, 20\n');
ashallow = c;
a = c;
c = 5.0 * c;
fprintf('   '); disp(c);

fprintf('Testing shallow copy (numeric arrays copy by value in MATLAB):\n');
fprintf('   '); disp(ashallow);

fprintf('Testing vector multiply, should be 0, -1, -2, -3, -4\n');
b = -ones(size(b));
b = b .* a;
fprintf('   '); disp(b);

fprintf('Testing vector norms using c = [0,5,10,15,20]\n');
fprintf('  2-norm      = %.4f\n', norm(c));
fprintf('  inf-norm    = %.4f\n', norm(c, inf));
fprintf('  one-norm    = %.4f\n', norm(c, 1));

% matrix demos
M = zeros(2,5);
M(1,:) = c;
M(2,:) = 3.0;
M = M + 2.0;

fprintf('Testing matrix infinity norm\n');
fprintf('   %.4f\n', norm(M, inf));
fprintf('Testing matrix one norm\n');
fprintf('   %.4f\n', norm(M, 1));
fprintf('Testing matrix two norm\n');
fprintf('   %.4f\n', norm(M, 2));

fprintf('Testing dot, should be 90\n');
a = [3, 3, 3, 3, 3];
c = [2, 4, 6, 8, 10];
fprintf('   %.4f\n', dot(a, c));

fprintf('Testing logspace, should be 0.01 0.1 1 10 100\n');
e = logspace(-2.0, 2.0, 5);
fprintf('   '); disp(e);

Y = zeros(10,5);
for i = 1:10
    Y(i,1) = i-1;
    Y(i,2) = -5.0 + (i-1);
    Y(i,3) = 2.0 + 2.0*(i-1);
    Y(i,4) = 20.0 - (i-1);
    Y(i,5) = -20.0 + (i-1);
end

Y4 = Y(:,5);
Y3 = Y(:,4);
Y4 = Y4 + Y3;
fprintf('Testing column extraction, should be all zeros:\n');
fprintf('   '); disp(Y4.');

d = linspace(0.0, 4.0, 5);
g = d + 2.0*e;
fprintf('Testing LinearSum, should be 0.02 1.2 4 23 204:\n');
fprintf('   '); disp(g);

d = d.^2;
fprintf('Testing power, should be 0 1 4 9 16:\n');
fprintf('   '); disp(d);

d = sqrt(d);
fprintf('Testing sqrt, should be 0 1 2 3 4:\n');
fprintf('   '); disp(d);

Z = [1,2,3;4,5,6];
fprintf('Testing transpose:\n');
fprintf(' Z:\n'); disp(Z);
fprintf(' Z'':\n'); disp(Z.');

fprintf('Testing matrix product, should be: 9 -1 9 -8 11 6\n');
A_ = eye(6);
A_(1,4) = 2.0;
A_(2,3) = -1.0;
A_(3,6) = 1.0;
A_(4,6) = -2.0;
A_(5,6) = 1.0;
xtrue_ = linspace(1.0, 6.0, 6).';
b_ = A_ * xtrue_;
fprintf('   '); disp(b_.');
end

function writeArray(A, filename)
    if exist('writematrix', 'file')
        writematrix(A, filename, 'Delimiter', 'tab');
    else
        dlmwrite(filename, A, 'delimiter', '\t', 'precision', 17);
    end
end

function A = readArray(filename)
    if exist('readmatrix', 'file')
        A = readmatrix(filename);
    else
        A = dlmread(filename, '\t');
    end
end
