function [part] = make_xval_partition(n, n_folds)
%PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% call upon a split in the original data using the part-splitted set of
% indices, e.g.
%trainingset_X = X(find(part == 2),:);
%trainingset_Y = Y(find(part == 2),:);

% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). 

fold_max_occupation = ceil(n/n_folds); %max number of examples in a fold
% e.g. n = 103, n_folds = 20, fold_max_occpuation = 6 // FMO

part_big = [ ] ; %ini
for i = 1:fold_max_occupation %for each iteration, append a random sorted vector with FMO entries
    part_big = horzcat(part_big,randperm(n_folds));
end
part = part_big(1:n); %now cut off all entries that are larger than the n you want. makes sure max difference between folds is 1.
party = part(randperm(length(part))); %shuffles... one more time...to increase randomness.
part = party; %output
end

