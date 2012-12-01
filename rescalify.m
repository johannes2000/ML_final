function [rescaled_matrix] = rescalify(data, miny, rangy)
% expects matrix = NxM, min = 1xM, range = 1xM, 
%cleverly subracts min and divides through range for every column
%keyboard
rescaled_matrix = (data - repmat(miny,size(data,1),1))*spdiags(1./(rangy)',0,size(data,2),size(data,2));
