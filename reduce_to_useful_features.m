function surviving_features = reduce_to_useful_features(full_lyrics, threshold)
% drops features that that are used less than the threshold of times
% overall, i.e. if threshold is 3 and a word was only used once by 
% examples 7 and 8 then it will not be returned by function

%takes in an N x M matrix of features, returns an N x K matrix of surviving
%features, where K is somewhere between 1 and M depending on the threhshold

total_feature_occurence = sum(full_lyrics,1); %returns an Nx1 row with the total coutns for each feature column
useful_features_idx = find(total_feature_occurence >= threshold);

%% output new matrix of useful words
surviving_features = full_lyrics(:,useful_features_idx);