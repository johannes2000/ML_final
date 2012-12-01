function ranks = make_final_prediction(model, example)
% Uses your trained model to make a final prediction for a SINGLE example.
%
% Usage:
%
%   RANKS = MAKE_FINAL_PREDICTION(MODEL, EXAMPLE);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes your trained model (output from INIT_MODEL) and a SINGLE 
% example, and returns a ranking ROW VECTOR as explained in the project
% overview.
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Your model should be loaded from disk in INIT_MODEL. DO NOT DO
% ANY TRAINING HERE.

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE
%%
% We only take in one example at a time.
X_lyr = make_lyrics_sparse(example, model.vocab);
X_aud = make_audio(example);

% load model ==> model_lyr.model, model_aud.model
%% rescale
Xt_audio_scaled = rescalify(X_aud, model_aud.min, model_aud.range);
Xt_lyrics_scaled = rescalify(X_lyr, model_lyr.min, model_lyr.range);
Yt = zeros(size(example,1)); %just a dummy vector because svmpredict needs that
clear X_aud;
clear X_lyr;
%% Lyrics and Audio prediction based on the models
[Yt_pred_lyr, ~, Yt_pred_prob_estimates_lyr] = svmpredict(Yt, Xt_lyrics_scaled, model_lyr.model, '-b 1');
[Yt_pred_aud, ~, Yt_pred_prob_estimates_aud] = svmpredict(Yt, Xt_audio_scaled, model_aud.model, '-b 1');

%% SINGLE CLASS OUTPUT
%combining the single-class predictions
Yt_pred_both_interact_binary = feature_operation_joined_binary_with_interaction(Yt_pred_aud, Yt_pred_lyr);

%%
%load level2 model
%load imputation order
[Yt_pred2, ~, Yt_pred2_prob_estimates] = svmpredict(Yt, Yt_pred_both_interact_binary, libsvmmodel2);
ranks = impute(Yt_pred2, imputation_order);
%{
% Find nearest neighbor
D = model.Xt*X';
[~,nn] = max(D);
yhat = model.Yt(nn);

% Convert into score vector
scores = zeros(1,10);
scores(yhat) = 1;

% Convert into ranks
ranks = get_ranks(scores);
%}
