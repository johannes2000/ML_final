% load the data
clear;
load ../data/music_dataset.mat

% make train and test partitions
rp = randperm(9704) > 9704*0.8;
train_indices = find(rp==0);
test_indices = find(rp==1);
ntrain = numel(train_indices);
ntest = numel(test_indices);
% shuffle the test/train indices
train_indices = train_indices(randperm(length(train_indices)));
test_indices = test_indices(randperm(length(test_indices)));

[Xtrain] = make_lyrics_sparse(train(train_indices), vocab);
[Xtest] = make_lyrics_sparse(train(test_indices), vocab);

Ytrain = zeros(ntrain, 1);
Ytest = zeros(ntest, 1);
for i=1:ntrain
  Ytrain(i) = genre_class( train( train_indices(i) ).genre );
end
for i=1:ntest
  Ytest(i) = genre_class( train( test_indices(i) ).genre );
end

yhats = kernel_regression_test(Xtrain,Ytrain,Xtest,2);
ranks = get_ranks(yhats);

test_acc = sum(ranks(:,1) == Ytest) / ntest
rank_loss = rankloss(ranks,Ytest)