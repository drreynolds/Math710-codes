function R = RK_stability_function(B, z)
    % Usage: R = RK_stability_function(B, z)
    %
    % Inputs:
    %   B is a Butcher table, with components:
    %      B.A -- the Butcher table matrix
    %      B.b -- the solution coefficients
    %   z is an array of points in the complex plane
    %
    % Outputs:
    %   R is an array of the same size as z, holding the values of the
    %     RK stability function
    %       R(z) = 1 + z * b^T inv(I-z*A) e
    %     at each entry of z.  The linear systems for all entries are
    %     solved together, as "pages" of s x s systems.  At an entry of
    %     z that lies exactly on a pole of R, R is not finite.
    %
    % Daniel R. Reynolds
    % Math & Stat @ UMBC

    % extract the components of the Butcher table
    A = double(B.A);
    b = double(B.b(:));
    s = numel(b);

    % solve (I - z*A) k = e at every entry of z at once
    zv = reshape(z, 1, 1, []);
    M = eye(s) - zv.*A;
    % (entries of z may land on a pole, where I - z*A is singular; there
    % pagemldivide returns non-finite values, so R is not finite)
    K = pagemldivide(M, ones(s, 1, numel(z)));
    R = 1 + z(:).*reshape(pagemtimes(b.', K), [], 1);
    R = reshape(R, size(z));
end
