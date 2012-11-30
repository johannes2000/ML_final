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
X = make_lyrics_sparse(example, model.vocab);
Xt_audio = make_audio(example);

% No labels for training example




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
