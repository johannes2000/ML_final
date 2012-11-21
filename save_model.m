% THIS IS NOT A FUNCTION.
%
% This is just a script that is called to train the model. It saves
% whatever we will need at test time.

% load('../data/music_dataset.mat');

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE

% Generate the sparse training set that we'll need for nearest neighbor
[Xt_lyrics] = make_lyrics_sparse(train, vocab);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end
Xt = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));

model.vocab = vocab;
model.Xt = Xt;
model.Yt = Yt;

save('my_model.mat', 'model');