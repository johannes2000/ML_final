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

%% Run algorithm

%{
Your predictions will be evaluated based on their mean reciprocal rank. Your code should produce an Nx10 matrix of ranks. Each row  is a ranking of the  genre labels in decreasing order of confidence for example . If the position of the true genre in your ranking vector is given by , the mean rank loss over your classifier’s predictions is:
%}

%keyboard;
%ranks_builtin_mod = predict_genre(Xt_lyrics, Xt_lyrics, Xt_audio, Xt_audio, Yt);
%loss_builtin = rankloss(ranks_builtin,Yt);
%loss_builtin_mod = rankloss(ranks_builtin_mod,Yt);
% the trainin error is .1824


%% built in on LYRICS
ranks_builtin_t = predict_genre(Xt_lyrics, Xt_lyrics, Xt_audio, Xt_audio, Yt);
ranks_builtin_q = predict_genre(Xt_lyrics, Xq_lyrics, Xt_audio, Xq_audio, Yt);
loss_builtin_t = rankloss(ranks_builtin_t,Yt) % the trainin error is .1824
save('-ascii', 'submit_testofformat.txt', 'ranks_builtin_q');

%%% RUN NB ON AUDIO
ranks_audio_nbt = classifier_NaiveBayes(Xt_audio, Yt); %testing
ranks_audio_nbq = classifierP_NaiveBayes(Xt_audio, Yt, Xq_audio); %training
rankloss(ranks_audio_nbt,Yt)  % the trainin error is .3348

%% VOTING
%% test different weights
upper_limit = 100;
loss_v_weight = zeros(upper_limit,2);
for weight = 1:upper_limit
    ranks_output = voting(ranks_builtin_t, ranks_audio_nbt, weight);
    loss_v_weight(weight,1) = weight;
    loss_v_weight(weight,2) = rankloss(ranks_output,Yt);
end
%% => five is the right weight
ranks_comb_test = voting(ranks_builtin_q, ranks_audio_nbq, 5);
save('-ascii', 'submit_1.txt', 'ranks_comb_test');



ranks = zeros(size(Xtrain,1),10); %this creates an Nx10 rank matrix\
posty = post; %posty gets stepwise depleted

for i = 1:10    
    [~,I] = max(posty,[],2); %this returns columns
    ranks(:,i) = I;
    for n= 1:size(Xtrain,1) %subtract
       posty(n,I(n)) = 0; 
    end
    %keyboard;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%11/16 6pm --- let's add the audio features -- DOESN'T WORK BECAUE OF THE
%%DIFF FEATURE SPACING, I THINK
Xt_comb = horzcat(Xt_lyrics, Xt_audio);
ranks_builtin_mod_comb = predict_genre(Xt_comb, Xt_comb, Xt_audio, Xt_audio, Yt);
loss_builtin_comb = rankloss(ranks_builtin_mod_comb,Yt)

%%%%%%%%%%%%%GNERALIZED LINEAR MODEL
%%
[ypred_GLM,ytest_GLM] = classifier_GeneralizedLinearModel(Xt_audio, Yt, Xq_audio)

%%
[ypred_GLM,ytest_GLM] = classifier_GeneralizedLinearModel(Xt_lyrics, Yt, Xq_lyrics)

%%
[ypred_GLM,ytest_GLM] = classifier_GeneralizedLinearModel(Xt_lyrics, Yt, Xq_lyrics)


%% DOESN"T WORK!!!!!! >> NEED TO RANKIFY AND MAKE SURE THERE IS ALWAYS ONE LEFT
%%rankify...conjoin 1-prediction with rank from NB

glmnb_combinedRank = zeros(size(ypred_GLM,1),1);
glmnb_combinedRank = ranks_audio;
glmnb_combinedRank(:,1) = round(ypred_GLM)
rankloss(glmnb_combinedRank,Yt)



%% join
%first, join test data and quiz data for both Audio adn lyrics
%STUB

%%split up data for crossvalidation
Xtrain = Xt_audio;
Ytrain = Yt;

%%NAIVEBAYES
ranks_audio = classifier_NaiveBayes(Xt_audio, Yt);
Xt_lyrics_full = full(Xt_lyrics);
ranks_lyrics = classifier_NaiveBayes(Xt_lyrics_full, Yt)


%%
ranks_freq = classifier_mereOutcomeFrequency(Yt);
loss_freq = rankloss(ranks_freq,Y)
%stub: combine preditions, train another NB with both of them

%%
loss = rankloss(ranks,Yt)

%% Save results to a text file for submission
save('-ascii', 'submit.txt', 'ranks');
save('gotlinmodel.mat')
