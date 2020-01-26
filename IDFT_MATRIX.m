% (C) 2014 Joseph Fernandez. Please see license.txt for license information.
%
function out = IDFT_MATRIX(N1, N2)

% form the phi matrix
% cell array to store the dft matrix
a = cell(1,N2);      % 1xN2 cell
[a{:}] = deal(conj(dftmtx(N1)));
phi = (1/N1)*blkdiag(a{:});
clear('a');

% now, form the psi matrix (function of m)
psi = zeros(N1,N1*N2);
% get the first row
for k2 = 0:N2-1
    psi(1,k2*N1+1:(k2+1)*N1) = [exp(1i*2*pi*k2/N2), zeros(1, N1-1)];
end
% fill out the matrix with shifted versions of this
for k1 = 1:N1-1
    psi(k1+1,:) = circshift(psi(1,:),[0,k1]);
end

% now form omega matrix
omega = zeros(N1*N2, N1*N2);
% first block - outside loop because 0^0 = 1
mask = psi == 0;
temppsi = psi.^0;
temppsi(mask) = 0;
omega(1:N1,:) = temppsi;
clear('temppsi');
for k1 = 1:N2-1
    omega(k1*N1+1:(k1+1)*N1, :) = psi.^k1;      % k1 is the power of m basically
end
omega = omega/N2;

% now, compute output matrix
out = omega*phi;
