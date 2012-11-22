[bestc_aud bestg_aud grid_audio g_params_aud c_params_aud] = svm_find_libsvm_parameters_v5_2_audio_only %find best audio params

string = 'bestc_aud bestg_aud g_params_aud c_params_aud';
string2 = [num2str(bestc_aud), num2str(bestg_aud), num2str(g_params_aud), num2str(c_params_aud)];
string3 =  [string, string2];
save('-ascii', 'results_audio.txt','string3');
save('results_audio.mat','string3');
save('-ascii', 'results_audio_grid.txt','grid_audio');
save('results_grid_audio.mat','grid_audio'); %all of this should really be in a struct

[bestc_lyr bestg_lyr grid_lyrics g_params_lyr c_params_lyr] = svm_find_libsvm_parameters_v5_2b_lyrics_only %find best lyric params
% predict 2 probability matrices
string = 'bestc_lyr bestg_lyr g_params_lyr c_params_lyr'
string2 = [num2str(bestc_lyr), num2str(bestg_lyr), num2str(g_params_lyr), num2str(c_params_lyr)]
string3 =  [string, string2]
save('-ascii', 'results_lyrics.txt','string3');
save('results_lyrics.mat','string3');
save('-ascii', 'results_lyrics_grid.txt','grid_lyrics');
save('results_grid_lyrics.mat','grid_lyrics');

[Yt_pred_prob_estimates_aud Yt_pred_prob_estimates_lyr Yq_pred_prob_estimates_aud Yq_pred_prob_estimates_lyr] = svm_predict_libsvm_lyrics_and_audio_v1(bestc_lyr, bestg_lyr, bestc_aud, bestg_aud)

%% this is from svm_predict_libsvm_ranking_from_lyrics_and_audio_probs_v2.m

%... from earlier: 
%{
%importantly, we now end up with Yt_pred_prob_estimates_aud and
%Yt_pred_prob_estimates_lyr on which we can train a classifier (set gamma
%and c through crossval as before, but optimizing for rank loss). Then,
%once we have these parameters, we can use the learned classifier to turn 
%Yq_pred_prob_estimates_lyr and Yq_pred_prob_estimates_aud into a final set
%of probability estimates, which we can rankify and submit.
%}
% coming in with Yt_pred_prob_estimates_aud and
%Yt_pred_prob_estimates_lyr, and Yq_pred_prob_estimates_lyr and Yq_pred_prob_estimates_aud

%% 1) vertcatting test and quiz sets 

training_size = size(Yt_pred_prob_estimates_lyr,1);

Ytq_pred_lyr = vertcat(Yt_pred_prob_estimates_lyr, Yq_pred_prob_estimates_lyr);
Ytq_pred_aud = vertcat(Yt_pred_prob_estimates_aud, Yq_pred_prob_estimates_aud);

%{ 
% for later
%% just some feature blow up
bam = horzcat(xd_aud, xd_aud.^2, sqrt(xd_aud), 2.^(xd_aud)); 
%}
%%self mult
%{
selfmult = bam; %the thing we would like to selfmultiply
k = size(selfmult,2); %number of features
output = selfmult;

for i = 1:k
    for j = i:k
        output = horzcat(output, selfmult(:,i).*selfmult(:,j)); %including every feature with itself
    end
end
%}

%% 2) horzcatting lyr and aud
Ytq_pred_both = horzcat(Ytq_pred_lyr, Ytq_pred_aud); %num of examples in testing set same for audio and lyrics

%% 3) adding interaction terms

Ytq_pred_both_interact = feature_operation_self_multiply(Ytq_pred_both);
%% 4) scaling as prep for SVM (through subtracting means and diving through the range)
% implemented here such that the scaling is the same for both quiz as well
% as training

data = Ytq_pred_both_interact;
Ytq_pred_both_interact_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));

%% 5) breaking (t+q) back up
%pred2 = Ytq_pred_both_interact_scaled
Yt_pred_both_interact_scaled = Ytq_pred_both_interact_scaled(1:training_size,:);
Yq_pred_both_interact_scaled = Ytq_pred_both_interact_scaled(training_size+1:end,:);

% => now we fit model for Yt_pred_both_interact_scaled, and apply to Yq_pred_both_interact_scaled

%% 6) finding a model that joins both predictions minimizing rank_loss
%ini 
Y = Yt;
X_scaled = Yt_pred_both_interact_scaled;

bestcv = 0;
c_params = [ ];
g_params = [ ];
grid = [ ];
rankloss_grid = [ ];
bestrankloss = 1;
g_idx = 0;
c_idx = 0;

%%

for log2c = -5:2:15, %%MOD
    c_idx = c_idx + 1; %look where you are in the c loop
    c_params(c_idx) = log2c;
    g_idx = 0; %every row...g_idx indexes column in grid
  for log2g = -15:2:3, %%MOD
    g_idx = g_idx + 1; %look where you are in the g loop
    g_params(g_idx) = log2g;
    cmd_cross_val = ['-t 2 -v 5 -h 0 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)]; %MOD
    cv = svmtrain(Y, X_scaled, cmd_cross_val);
    if (cv >= bestcv), %outputs the accuracy
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    grid(c_idx, g_idx) = cv;
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % this actually creates model to predict, and gets the rank error
    cmd_predict = ['-t 2 -b 1 -h 0 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)]; %MOD
    libsvmmodel2 = svmtrain(Y, X_scaled, cmd_predict);
    % use to predict on same dataset
    % SHOULD IMPLEMENT CROSS VALIDATION HERE Too
    [Yt_pred2, Yt_pred2_accuracy, Yt_pred2_prob_estimates] = svmpredict(Yt, X_scaled, libsvmmodel2, '-b 1');
    Yt_pred2_ranks = get_ranks(Yt_pred2_prob_estimates);
    ranklossy = rankloss(Yt_pred2_ranks, Y);
    rankloss_grid(c_idx, g_idx) = ranklossy; %make me a rankloss grid
    % see if it's the best rankloss
    if (ranklossy <= bestrankloss), 
      bestrankloss = ranklossy; bestc_rl = 2^log2c; bestg_rl = 2^log2g;
      bestmodel = libsvmmodel2;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
  end
end

%%
save('gotallthewaytoend.mat')
contourf(grid);
g_params
c_params'
grid
bestrankloss
rankloss_grid

%% 7) final prediction of ranks

rand_Yq = zeros(size(Yq_pred_both_interact_scaled,1),1);
[Yq_pred2, ~, Yq_pred2_prob_estimates] = svmpredict(rand_Yq, Yq_pred_both_interact_scaled, bestmodel, '-b 1');

final_ranks = get_ranks(Yq_pred2_prob_estimates);
save('-ascii', 'ranks_final_cv_la_t2.txt','final_ranks');
save('gotallthewaytoend2.mat')