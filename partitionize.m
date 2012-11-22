function [Xtrain XDev Ytrain YDev] = partitionize(X, Y, number_of_partitions)
% splits the training set up say 4 ways, makes 3 of them Xtest/Ytest, and
% the rest the "testing set" within the original training set

n = size(X,1);
part = make_xval_partition(n, number_of_partitions);

%make the first random partition the Dev set
XDev = X(find(part == 1),:);
YDev = Y(find(part == 1),:);

%put the others together and make them the train-train set
Xtrain = [ ];
Ytrain = [ ];
for i = 2:number_of_partitions
    Xtrain = vertcat(Xtrain, X(find(part == i),:));
    Ytrain = vertcat(Ytrain, Y(find(part == i),:));
end
