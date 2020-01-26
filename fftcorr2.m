% (C) 2014 Joseph Fernandez. Please see license.txt for license information.
%
% Joseph Fernandez 2011
% 
% This function will take a 2d correlation filter (y,x) and an image
% matrix (y,x) and will return the 2-dimensional correlation. The output
% is determined by use of the fast fourier transform (FFT) for
% implementation speed. The output is the same size as image.

function out = fftcorr2(filter, im)

% final dimensions of output
dim_row = size(im,1)+size(filter,1)-1;
dim_col = size(im,2)+size(filter,2)-1;

% fft size we use (use powers of 2 for speed)
fft_row = 2^nextpow2(dim_row);
fft_col = 2^nextpow2(dim_col);

% get fft of image matrix
movie_fft = fft2(im, fft_row, fft_col);

% take conjugate of filter - equivalent to a flip in video domain
filter_conj = flipdim(flipdim(filter,1),2);
filter_fft = fft2(filter_conj, fft_row, fft_col);

product_fft = movie_fft.*filter_fft;

outfft = ifft2(product_fft);

out = outfft(1:dim_row, 1:dim_col);

