clear;
load ../data/music_dataset.mat;
% using the matlab implementation of naive bayes

%make the lyrics
X_s = make_lyrics_sparse(train, vocab);
%try making it a non-sparse matrix
X = full(X_s);

rp = randperm(9704) > 9704*0.8;
train_indices = find(rp==0);
test_indices = find(rp==1);
ntrain = numel(train_indices);
ntest = numel(test_indices);
%shuffle the test/train indices
train_indices = train_indices(randperm(length(train_indices)));
test_indices = test_indices(randperm(length(test_indices)));

Xtrain = X(train_indices,:);
Xtest = X(test_indices,:);

Ytrain = cell(ntrain, 1);
Ytest = cell(ntest, 1);
Ytest_numeric = zeros(ntest, 1);
for i=1:ntrain
  Ytrain{i} = train( train_indices(i) ).genre;
end
for i=1:ntest
  Ytest{i} = train( test_indices(i) ).genre;
  Ytest_numeric(i) = genre_class( train( test_indices(i) ).genre );
end

nb = NaiveBayes.fit(Xtrain,Ytrain,'Distribution','mvmn');
ranks = get_ranks( posterior(nb,Xtest) );

test_acc = (sum( ranks(:,1) == Ytest_numeric ) / ntrain) * 100
rank_loss = rankloss(ranks,Ytest_numeric)