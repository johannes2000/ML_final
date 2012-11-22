function [Yt_pred_prob_estimates_aud Yt_pred_prob_estimates_lyr Yq_pred_prob_estimates_aud Yq_pred_prob_estimates_lyr] = svm_predict_libsvm_lyrics_and_audio_v1(lyrics_c, lyrics_g, audio_c, audio_g)

%% you come in with the parameters for the two SVM's
%% for actual prediction

%lyrics_c = 5; %TBD
%lyrics_g = -9; %TBD

%audio_c = 10;
%audio_g = -8;


%% load data
%clear;
load ../data/music_dataset.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);


%% scaling as prep for SVM (through subtracting means and diving through the range)
% implemented her such that the scaling is the same for both quiz as well
% as training
Xt_length = size(Xt_lyrics,1); %num of examples in testing set same for audio and lyrics

% scaling all lyrics
Xt_Xq_lyrics = vertcat(Xt_lyrics, Xq_lyrics);
data = Xt_Xq_lyrics ;
Xt_Xq_lyrics_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
% breaking lyrics back up
Xt_lyrics_scaled = Xt_Xq_lyrics_scaled(1:Xt_length,:);
Xq_lyrics_scaled = Xt_Xq_lyrics_scaled(Xt_length+1:end,:); %check: mat dim look right

% scaling all audio
Xt_Xq_audio = vertcat(Xt_audio, Xq_audio);
data = Xt_Xq_audio ;
Xt_Xq_audio_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
% breaking audio back up
Xt_audio_scaled = Xt_Xq_audio_scaled(1:Xt_length,:);
Xq_audio_scaled = Xt_Xq_audio_scaled(Xt_length+1:end,:);




%% predicting lyrics
log2c = lyrics_c;
log2g = lyrics_g;

Yq_rand = zeros(size(Xq_lyrics_scaled,1),1); %any random vect will do here

cmd = ['-t 2 -b 1 -h 0 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
libsvmmodel_lyr = svmtrain(Yt, Xt_lyrics_scaled, cmd); %learn model on test set %takes about 10 min or so
%this predicts on the testing set the model was trained on.
[Yt_pred_lyr, Yt_pred_accuracy_lyr, Yt_pred_prob_estimates_lyr] = svmpredict(Yt, Xt_lyrics_scaled, libsvmmodel_lyr, '-b 1');
%and on the quiz set.
[Yq_pred_lyr, Yq_pred_accuracy_lyr, Yq_pred_prob_estimates_lyr] = svmpredict(Yq_rand, Xq_lyrics_scaled, libsvmmodel_lyr, '-b 1');
%accuracy is meaningless, because Yq_rand is


%% predicting audio
log2c = audio_c;
log2g = audio_g;

Yq_rand = zeros(size(Xq_lyrics_scaled,1),1); %any random vect will do here

cmd = ['-t 2 -b 1 -h 0 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
libsvmmodel_aud = svmtrain(Yt, Xt_audio_scaled, cmd); %learn model on test set %takes 3 min
%this predicts on the testing set the model was trained on.
[Yt_pred_aud, Yt_pred_accuracy_aud, Yt_pred_prob_estimates_aud] = svmpredict(Yt, Xt_audio_scaled, libsvmmodel_aud, '-b 1'); %64.92% accuracy
%and on the quiz set.
[Yq_pred_aud, Yq_pred_accuracy_aud, Yq_pred_prob_estimates_aud] = svmpredict(Yq_rand, Xq_audio_scaled, libsvmmodel_aud, '-b 1');
%accuracy is meaningless, because Yq_rand is

%%
%importantly, we now end up with Yt_pred_prob_estimates_aud and
%Yt_pred_prob_estimates_lyr on which we can train a classifier (set gamma
%and c through crossval as before, but optimizing for rank loss). Then,
%once we have these parameters, we can use the learned classifier to turn 
%Yq_pred_prob_estimates_lyr and Yq_pred_prob_estimates_aud into a final set
%of probability estimates, which we can rankify and submit.


