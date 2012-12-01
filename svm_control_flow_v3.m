%{

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

%}


%% this stuff ran through, with these results:
bestc_lyr = 2^7;
bestg_lyr = 2^(-9);
bestc_aud = 2^16;
bestg_aud = 2^(-6);

%%
[Yt_pred_prob_estimates_aud Yt_pred_prob_estimates_lyr Yq_pred_prob_estimates_aud Yq_pred_prob_estimates_lyr] = svm_predict_libsvm_lyrics_and_audio_v1(bestc_lyr, bestg_lyr, bestc_aud, bestg_aud)

%turns out the pred_prob_estimates_* are shit and don't reflec the choice
%that really goes into the prediction (
%}

%%
%save('simplepredictions_aud_lyr_t_q.mat', 'Yt_pred_aud', 'Yt_pred_lyr', 'Yq_pred_aud', 'Yq_pred_lyr');
load('simplepredictions_aud_lyr_t_q.mat')

%% 0) loading everything into this function
load ../data/music_dataset.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);


%% check
%accuracy
sum(Yt_pred_aud == Yt)/numel(Yt) %70.39%
sum(Yt_pred_lyr == Yt)/numel(Yt) %96.15%


%{
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%



%% NOT USED RIGHT NOW
% get the ones that are not correct and make mere collective frequency imputation

%calc order of leftovers ******** AUDIO
%{
Yt_pred_aud_wrong = find(Yt_pred_aud~=Yt);
edges = [.5:1:10.5];
[a,~] = histc(Yt(Yt_pred_aud_wrong),edges);
[~,imputation_order] = sort(a,'descend');
%}
imputation_order = impute_make_order_based_on_mistakes(Yt_pred_aud, Yt)
Yt_pred_aud_imputed = impute(Yt_pred_aud, imputation_order);
%sum(Yt~=Yt_pred_aud)/numel(Yt)
%rankloss(Yt_pred_aud_imputed,Yt)  %upper bound on below
%==> 
Yq_pred_aud_imputed = impute(Yq_pred_aud, imputation_order(1:10));

%calc order of leftover ********** LYRICS
%{
Yt_pred_lyr_wrong = find(Yt_pred_lyr~=Yt);
edges = [.5:1:10.5];
[a,~] = histc(Yt(Yt_pred_lyr_wrong),edges);
[~,imputation_order] = sort(a,'descend');
%}

imputation_order = impute_make_order_based_on_mistakes(Yt_pred_lyr, Yt)
Yt_pred_lyr_imputed = impute(Yt_pred_lyr, imputation_order);
%sum(Yt~=Yt_pred_lyr)/numel(Yt) %upper bound on below
%rankloss(Yt_pred_lyr_imputed,Yt)
% ==>
Yq_pred_lyr_imputed = impute(Yq_pred_lyr, imputation_order(1:10));
%%
com

%% NOT USED RIGHT NOW
%% temp fix -- gotten from prediction routine manugally, must adjust to use permanantly
Yt_pred_prob_estimates_aud = Yt_pred_aud_imputed;
Yt_pred_prob_estimates_lyr = Yt_pred_lyr_imputed;
Yq_pred_prob_estimates_aud = Yq_pred_aud_imputed;
Yq_pred_prob_estimates_lyr = Yq_pred_lyr_imputed;


%% NOT USED RIGHT NOW
%% scaling 1) vertcatting test and quiz sets 

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

%% NOT USED RIGHT NOW
%% scaling 2) horzcatting lyr and aud
Ytq_pred_both = horzcat(Ytq_pred_lyr, Ytq_pred_aud); %num of examples in testing set same for audio and lyrics
%% NOT USED RIGHT NOW
%% scaling 3) adding interaction terms
Ytq_pred_both_interact = feature_operation_self_multiply(Ytq_pred_both);
%% NOT USED RIGHT NOW
%% scaling 4) scaling as prep for SVM (through subtracting means and diving through the range)
% implemented here such that the scaling is the same for both quiz as well
% as training
data = Ytq_pred_both_interact;
Ytq_pred_both_interact_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
%% NOT USED RIGHT NOW
%% scaling 5) breaking (t+q) back up
%pred2 = Ytq_pred_both_interact_scaled
Yt_pred_both_interact_scaled = Ytq_pred_both_interact_scaled(1:training_size,:);
Yq_pred_both_interact_scaled = Ytq_pred_both_interact_scaled(training_size+1:end,:);
% => now we fit model for Yt_pred_both_interact_scaled, and apply to Yq_pred_both_interact_scaled



%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%
%}

%% 5b -- using single class output suggestions from two classifiers
%instaed uses 'Yt_pred_aud', 'Yt_pred_lyr', 'Yq_pred_aud', 'Yq_pred_lyr'

%not really scaled, but legacy
Yt_pred_both_interact_binary = feature_operation_joined_binary_with_interaction(Yt_pred_aud, Yt_pred_lyr);
Yq_pred_both_interact_binary = feature_operation_joined_binary_with_interaction(Yq_pred_aud, Yq_pred_lyr);


%% 6) finding a model that joins both predictions minimizing rank_loss
%ini 


%% FOR PROBABILITY MATRICES - RBF
%{

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
%}
%% FOR BINARY INTERACTION MATRIX BETWEEN SINGLE OUTPUT - LINEAR
Y = Yt;
%X_scaled = Yt_pred_both_interact_scaled;
X_scaled = Yt_pred_both_interact_binary;

bestcv = 0;
c_params = [ ];
g_params = [ ];
grid = [ ];
rankloss_grid = [ ];
bestrankloss = 1;
bestimputationorder = [ ];
g_idx = 0;
c_idx = 0;

%%
for log2c = -5:2:25, %%MOD (-5 to 15??)
    %keyboard;
    c_idx = c_idx + 1; %look where you are in the c loop
    c_params(c_idx) = log2c;
    %cmd_cross_val = ['-t 0 -v 4 -h 0 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)]; %MOD
    cmd_cross_val = ['-t 0 -v 4 -h 0 -c ', num2str(2^log2c)]; %MOD
    cv = svmtrain(Y, X_scaled, cmd_cross_val);
    if (cv >= bestcv), %outputs the accuracy
      bestcv = cv; bestc = 2^log2c; % bestg = 2^log2g;
    end
    grid(c_idx) = cv;
    fprintf('%g %g (best c=%g, bestcv=%g)\n', log2c, cv, bestc, bestcv);
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % this actually creates model to predict, and gets the rank error
    cmd_predict = ['-t 0 -c ', num2str(2^log2c)]; %MOD
    libsvmmodel2 = svmtrain(Y, X_scaled, cmd_predict);
    % use to predict on same dataset
    % SHOULD IMPLEMENT CROSS VALIDATION HERE Too
    [Yt_pred2, Yt_pred2_accuracy, Yt_pred2_prob_estimates] = svmpredict(Yt, X_scaled, libsvmmodel2);
    %now we must imnprovise ranks FORMER %Yt_pred2_ranks = get_ranks(Yt_pred2_prob_estimates);
    %%% IMPROVISE RANKS %%%%
    imputation_order = impute_make_order_based_on_mistakes(Yt_pred2, Yt);
    Yt_pred2_ranks = impute(Yt_pred2, imputation_order);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ranklossy = rankloss(Yt_pred2_ranks, Y);
    rankloss_grid(c_idx) = ranklossy; %make me a rankloss grid
    % see if it's the best rankloss
    if (ranklossy <= bestrankloss), 
      bestrankloss = ranklossy; 
      bestc_rl = 2^log2c;
      bestmodel = libsvmmodel2;
      bestimputationorder = imputation_order;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
  end
end

%%
save('gotallthewaytoend_usingBestSingle_CgridLinear_96.3.mat')
%contourf(grid);
%g_params
c_params'
grid
bestrankloss
rankloss_grid

%% 7) final prediction of ranks
%{
rand_Yq = zeros(size(Yq_pred_both_interact_scaled,1),1);
[Yq_pred2, ~, Yq_pred2_prob_estimates] = svmpredict(rand_Yq, Yq_pred_both_interact_scaled, bestmodel, '-b 1');

final_ranks = get_ranks(Yq_pred2_prob_estimates);
save('-ascii', 'ranks_final_cv_la_t2.txt','final_ranks');
save('gotallthewaytoend2.mat')
%}

%% 7b) final prediction of ranks on the basis of single rank prediction in subclassifiers

rand_Yt = zeros(size(Yt_pred_both_interact_binary,1),1);
rand_Yq = zeros(size(Yq_pred_both_interact_binary,1),1);
[Yt_pred2f, ~, Yt_pred2f_prob_estimates] = svmpredict(Yt, Yt_pred_both_interact_binary, bestmodel);
[Yq_pred2f, ~, Yq_pred2f_prob_estimates] = svmpredict(rand_Yq, Yq_pred_both_interact_binary, bestmodel);

imputation_order = impute_make_order_based_on_mistakes(Yt_pred2f, Yt);
Yt_pred2f_ranks = impute(Yt_pred2f, imputation_order);
Yq_pred2f_ranks = impute(Yq_pred2f, imputation_order);
rankloss(Yt_pred2f_ranks, Yt)

%%
%final_ranks = get_ranks(Yq_pred2_prob_estimates);
save('-ascii', 'Yq_pred2f_ranks_binary_noProb.txt','Yq_pred2f_ranks');
save('gotallthewaytoend2.mat')