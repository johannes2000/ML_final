function ranks = predict_genre(Xt_lyrics, Xq_lyrics, ...
                               Xt_audio, Xq_audio, ...
                               Yt)
% Returns the predicted rankings, given lyric and audio features.
%
% Usage:
%
%   RANKS = PREDICT_GENRE(XT_LYRICS, YT_LYRICS, XQ_LYRICS, ...
%                         XT_AUDIO, YT_AUDIO, XQ_AUDIO);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of lyric and audio features and produces a
% ranking matrix as explained in the project overview. 
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

%%
% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE
N = size(Xq_lyrics, 1);
scores = zeros(N, 5);
Xt = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));
Xq = bsxfun(@rdivide, Xq_lyrics, sum(Xq_lyrics, 2));

D = Xq*Xt';
[~, idx] = max(D, [], 2); %gives a vector, giving maxcolum  for every example
ynn = idx(:, 1);
yhat = Yt(ynn);

for i=1:N
    scores(i, yhat(i)) = 1;
end
ranks_lyrics = get_ranks(scores);

%%% RUN NB ON AUDIO

ranks_audio_nbq = classifierP_NaiveBayes(Xt_audio, Yt, Xq_audio);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% RUNNING NB to comine the two
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ranks_comb_X = horzcat(ranks_lyrics, ranks_audio_nbq)

nb_ranksComb = NaiveBayes.fit(ranks_comb, Yt)
[post_t,Ypred] = posterior(nb,ranks_comb);


%%%%%%%%%%TRAINING ERROR

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


%%%%%%%%%TESTING ERROR

[post_q,Ypred] = posterior(nb,Xtrain);



end
