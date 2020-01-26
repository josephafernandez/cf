% Copyright 2008 Carnegie Mellon University
% Author: Andres Rodriguez
% Date: Mar 19, 2009
%
% Description: This function returns the indices of a matrix
%  ex mat=[ 1  4  7  10 ;
%           2  5  8  11 ;
%           3  6  9  12 ]
% [3 4] = size(A)
% returns ind = [ 1 1 ;
%                 2 1 ;
%                 3 1 ;
%                 1 2 ;
%                 2 2 ;
%                 3 2 ;
%                 1 3 ;
%                 2 3 ;
%                 3 3 ;
%                 1 4 ;
%                 2 4 ;
%                 3 4  ]
%
% function ind = getMatIndices(mat_size)
%

function ind = getMatIndices(mat_size)

A = ones(mat_size);
[r,c] = find(A==1);
ind = [r,c];
