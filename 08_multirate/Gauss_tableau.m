function B = Gauss_tableau(sF)
    % Usage: B = Gauss_tableau(sF)
    %
    %        Returns the Butcher table of the sF-stage Gauss--Legendre method, of
    %        order 2*sF, in double precision, as a struct with fields 'A',
    %        'b' and 'c'.  The nodes c are the Gauss points on [0,1], and as a
    %        collocation method its coefficients are
    %          b_j = int_0^1 l_j(tau) dtau,  a_ij = int_0^{c_i} l_j(tau) dtau,
    %        where l_j is the Lagrange basis polynomial for the nodes c.  The b_j
    %        are the Gauss weights themselves, and since each l_j has degree
    %        sF-1, a_ij is computed exactly by the same Gauss rule mapped to
    %        [0,c_i].  (Solving the equivalent collocation conditions with the
    %        Vandermonde matrix of the nodes would lose accuracy as sF grows,
    %        since that matrix is ill-conditioned.)
    %
    %        MATLAB has no built-in Gauss--Legendre rule, so the Gauss points x
    %        and weights w on [-1,1] are computed with the Golub--Welsch
    %        algorithm: x are the eigenvalues of the symmetric tridiagonal
    %        Jacobi matrix of the Legendre polynomials, whose off-diagonal
    %        entries are k/sqrt(4k^2-1), k = 1,...,sF-1, and w_j = 2 v_{1j}^2,
    %        where v_j is the corresponding unit eigenvector.
    %
% Function to construct the Butcher table of a Gauss--Legendre method, for
% use as the fast method in MRI_order.
%
% Daniel R. Reynolds
% Math & Stat @ UMBC
%
k = (1:sF-1).';
offdiag = k./sqrt(4*k.^2 - 1);
[V, D] = eig(diag(offdiag, 1) + diag(offdiag, -1));
[x, idx] = sort(diag(D));
w = 2*V(1,idx).'.^2;
c = (x + 1)/2;
b = w/2;

A = zeros(sF, sF);
for i = 1:sF
    tau = c(i)*c;                     % Gauss points mapped to [0,c_i]
    for j = 1:sF
        A(i,j) = c(i)*dot(b, lagrange(j, tau));
    end
end
B = struct('A', A, 'b', b, 'c', c);

    % utility routine to evaluate l_j at the points tau
    function lj = lagrange(j, tau)
        lj = ones(size(tau));
        for m = 1:sF
            if (m ~= j)
                lj = lj.*((tau - c(m))/(c(j) - c(m)));
            end
        end
    end
end
