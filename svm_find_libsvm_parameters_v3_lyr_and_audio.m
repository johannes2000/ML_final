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

% subsection of dataset
%use to randomly pick 1/50th of the sample
[xt_lyr xt_aud xd_lyr xd_aud yt yd] = partitionize2(Xt_lyrics, Xt_audio, Yt, 50);

%% remove bad features
xd_lyr_surv = reduce_to_useful_features(xd_lyr, 3);
xd_lyr = xd_lyr_surv;

%%
%append audio
xd_lyr_aud = horzcat(xd_aud, xd_lyr); %doublecheck
%scale
data = xd_lyr_aud;

%SCALING
xd_lyr_aud_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));


%% only audio
%scale
data = xd_aud;
xd_lyr_aud_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));


%%
%ini 
Y = yd;
X_scaled = xd_lyr_aud_scaled;

bestcv = 0;
c_params = [ ];
g_params = [ ];
grid = [ ];
g_idx = 0;
c_idx = 0;

for log2c = -5:2:13,
    c_idx = c_idx + 1; %look where you are in the c loop
    c_params(c_idx) = log2c;
    g_idx = 0; %every row...g_idx indexes column in grid
  for log2g = -13:2:3,
    g_idx = g_idx + 1; %look where you are in the g loop
    g_params(g_idx) = log2g;
    cmd = ['-t 0 -v 4 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
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
contour(grid);
g_params
c_params'
bestcv
grid
