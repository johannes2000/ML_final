clear;
load ../data/music_dataset.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);


%%conjoin audio training and test set

full_lyrics = vertcat(Xt_lyrics, Xq_lyrics);

%%drop words (language features that are less than threshold)
full_lyrics = Xt_lyrics;

threshold = 8;
total_feature_occurence = sum(full_lyrics,1);
useful_words_idx = find(total_feature_occurence >= threshold);

%% limit training set to useful words
Xt_lyrics_useful = Xt_lyrics(:,useful_words_idx);
%% append audio features to training set
Xt_useful_lyrics_audio = horzcat(Xt_lyrics_useful, Xt_audio);

%Xt_useful_lyrics_audio = full(Xt_useful_lyrics_audio);
%%% HERE NOW IS LOGIC FROM classifier_NativeBayes
nb = NaiveBayes.fit(Xt_useful_lyrics_audio, Yt, 'Distribution','kernel')
%Ypred = nb.predict(nb, Xtest) %this is an Nx1 matrix
[post_t,Ypred] = posterior(nb,Xt_useful_lyrics_audio);
%Ypred is the predicted class...first in rank-output
%post is a N-by-nb.nclasses matrix containing the posterior probability of
%each observation for each class. post(i,j) is the posterior probability of 
%point/example I belonging to class j. 

%%
Xt_useful_lyrics_audio_sparse = sparse(Xt_useful_lyrics_audio);

save
%%
%keyboard;
ranks = zeros(size(Xtrain,1),10); %this creates an Nx10 rank matrix
%posty gets stepwise depleted
posty = post_t;

for i = 1:10    
    %find the index of the largest probablity in that nx10 post array of
    %probabilities
    [~,I] = max(posty,[],2); %this returns columns
    %for all examples delete cases we just copied over
    %assign that feature to the i'th column in the rank matrix
    ranks(:,i) = I;
    %now subtract thsi from posty
    for n= 1:size(Xtrain,1)
       posty(n,I(n)) = 0; 
    end
    %keyboard;
end


%%

[post_q,Ypred] = posterior(nb,Xtrain);

