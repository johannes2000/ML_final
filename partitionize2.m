function [X1train X2train X1Dev X2Dev Y1train YDev] = partitionize2(X1, X2, Y, number_of_partitions)
% splits the training set up say 4 ways, makes 3 of them Xtest/Ytest, and
% the rest the "testing set" within the original training set

n = size(X1,1);
part = make_xval_partition(n, number_of_partitions);

%make the first random partition the Dev set
X1Dev = X1(find(part == 1),:);
X2Dev = X2(find(part == 1),:);
YDev = Y(find(part == 1),:);

%put the others together and make them the train-train set
X1train = [ ];
X2train = [ ];
Y1train = [ ];
for i = 2:number_of_partitions
    X1train = vertcat(X1train, X1(find(part == i),:));
    X2train = vertcat(X2train, X2(find(part == i),:));
    Y1train = vertcat(Y1train, Y(find(part == i),:));
end
