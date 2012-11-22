function [bestc_lyr bestg_lyr grid_lyrics g_params_lyr c_params_lyr] = svm_find_libsvm_parameters_v5_2b_lyrics_only;
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


% subsection of dataset with 2
%use to randomly pick 1/50th of the sample

%HERE: No partitioning
%[xt_lyr xt_aud xd_lyr xd_aud yt yd] = partitionize2(Xt_lyrics, Xt_audio, Yt, 10);

%% should do feature occurance calc on both test and quiz

%{
%% remove bad features
Xt_lyr_surv = reduce_to_useful_features(Xt_lyrics, 3);
%xd_lyr = xd_lyr_surv;
%}

%{
%% append audio FIRST TO SURVIVING
Xt_lyrics_surv_audio = horzcat(Xt_audio, Xt_lyr_surv); 

% ==> NOW subsection of dataset OLD -- just with one
[xt_lsa xd_lsa yt yd] = partitionize(Xt_lyrics_surv_audio, Yt, 2);

%SCALING
data = xd_lsa;
xd_lsa_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
%}

%{
%append audio
xd_lyr_aud = horzcat(xd_aud, xd_lyr); %doublecheck
%scale
data = xd_lyr_aud;
%}

%%%%%%%%%%%%%%%%%%%%%%%% ONLY LYRICS %%%%%%%%%%%%%%%%%%%%%%%%

% if one wants to limit features
%Xt_lyrics = reduce_to_useful_features(Xt_lyrics, 3)


%% %scale
data = Xt_lyrics;
Xt_lyrics_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));


%%
%ini 
Y = Yt;
X_scaled = Xt_lyrics_scaled;

bestcv = 0;
c_params = [ ];
g_params = [ ];
grid = [ ];
g_idx = 0;
c_idx = 0;

for log2c = -1:1:11,
    c_idx = c_idx + 1; %look where you are in the c loop
    c_params(c_idx) = log2c;
    g_idx = 0; %every row...g_idx indexes column in grid
  for log2g = -15:1:-3,
    g_idx = g_idx + 1; %look where you are in the g loop
    g_params(g_idx) = log2g;
    cmd = ['-t 2 -h 0 -v 4 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    %cmd = ['-v 4 -c ', num2str(2^-2), ' -g ', num2str(2^-5)];
    cv = svmtrain(Y, X_scaled, cmd);
    %[yd_lyr_pred, accuracy, prob_estimates] =  svmpredict(yd_lyr, xd_lyr, libsvmmodel, '-b 1');
    if (cv >= bestcv), %outputs the accuracy
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    grid(c_idx, g_idx) = cv;
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end
contourf(grid);
bestc_lyr = bestc
bestg_lyr = bestg
g_params_lyr = g_params
c_params_lyr = c_params
bestcv_lyr = bestcv
grid_lyrics = grid
