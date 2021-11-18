%% Input data

load('DBM_sample_data.mat')

% body_parts_X = your_X_data_here;
% body_parts_Y = your_Y_data_here;
% behavior_pseudolabels = your_pseudolabels_here

% body_parts_X and body_parts_Y should each be formatted as a cell array,
% where each cell contains a PxT matrix, where P is the number of tracked
% body parts, and T is the number of timesteps per sequence.

% behavior_pseudolabels should be formatted as a cell array, where each
% cell contains a 1xT categorical array, with labels corresponding to the
% experimenter-defined pseudolabel categories.



% Validate formatting of pose sequences and pseudolabels
assert(iscell(body_parts_X), 'tracking X data must be formatted as cell array')
assert(iscell(body_parts_Y), 'tracking Y data must be formatted as cell array')
assert(iscell(behavior_pseudolabels), 'behavior pseudolabels must be formatted as cell array')
assert(all(cellfun(@iscategorical, behavior_pseudolabels)))
assert(numel(unique(cellfun(@(X) string(strjoin(categories(X),'')), behavior_pseudolabels)))==1, ...
    'Pseudolabel arrays do not all have the same set of categories')
assert(numel(body_parts_X)==numel(body_parts_Y), 'Size of X and Y tracking data should be identical')
assert(all(cellfun(@(X,Y) all(size(X)==size(Y)), body_parts_X, body_parts_Y)), 'Size of X and Y tracking data should be identical')
assert(numel(body_parts_X)==numel(behavior_pseudolabels), 'Number of pose sequences and pseudolabel sequences must be equal')
assert(all(cellfun(@(X) size(X,2), body_parts_X)==cellfun(@(X) size(X,2), behavior_pseudolabels)), ...
    'Length (dimension 2) of each pseudolabel sequence must match the length of the corresponding tracking data sequence')

%% Format X/Y tracking data into pose sequences

pose_sequences = dbmFormatInputs(body_parts_X, body_parts_Y);

%% Train DBM model

N = numel(pose_sequences);

use_gpu = false;
val_fraction = 0.1; % Proportion of input sequences to be held back as validation data

val_ind = false(N,1);
val_ind(randperm(N, round(N*val_fraction))) = true;


[dbm_mdl, dbm_train_report] = dbmTrainModel(pose_sequences(~val_ind), behavior_pseudolabels(~val_ind), ... % training data
    pose_sequences(val_ind), behavior_pseudolabels(val_ind), use_gpu); % validation data

% dbm_mdl = the trained DBM network, which can be used to predict
%   behavioral pseudolabels and/or map pose sequences to the latent space

% dbm_train_report = summary data about how the model performed
%   during the training process.
%%

[pseudolabel_predictions, latent_trajectories] = dbmGetModelOutputs( dbm_mdl, pose_sequences, use_gpu );

% pseudolabel_predictions = model's prediction of original training targets
% latent trajectories = set of N (default = 10) variables mapping subject's
%   behavior to a point in the "behavior space" learned by the model

%% Extract microstates using k-means algorithm

num_microstates = 50;
max_iter = 100;
replicates = 1;
use_parallel = false;

% Extract microstates
[microstate_labels, microstate_centroids] = dbmExtractMicrostates(latent_trajectories, num_microstates, max_iter, replicates);

% Re-numebr microstates according to which pseudolabel category they
% overlap with most
pseudolabel_cats = categories(behavior_pseudolabels{1});
microstates = [1:num_microstates];
label_overlap = (cat(2,behavior_pseudolabels{:})==pseudolabel_cats) * (cat(2,microstate_labels{:}).'==microstates);
[~,max_overlap] = max(label_overlap,[],1);
[~,sort_ind] = sortrows([max_overlap; sum(label_overlap,1)].',[1 2]);
microstates = microstates(sort_ind);
label_overlap = label_overlap(:,sort_ind);
[~,microstate_labels] = cellfun(@(X) ismember(X,microstates), microstate_labels, 'UniformOutput', false);

% Normalize overlap matrix column-wise
label_overlap = round(1000.*(label_overlap./sum(label_overlap,1)))./10;



%% Visualization

figure(1)
clf

h = heatmap(label_overlap);
h.XDisplayLabels = string(microstates);
h.YDisplayLabels = pseudolabel_cats;
title({'% pseudolabel category by microstate'; '(columns sum to ~100%)'})
xlabel('Microstate #')
ylabel('Behavior pseudolabel category')


figure(2)
clf

imagesc(cat(1,microstate_labels{:}))
colormap(jet(num_microstates))
cb = colorbar;
ticks = linspace(cb.Limits(1), cb.Limits(2), num_microstates+1);
ticks = ticks(1:end-1) + mean(diff(ticks))/2;
cb.Ticks = ticks;
cb.TickLabels = 1:num_microstates;
ylabel('Trials')
xlabel('Video frames')
title('Microstate occurrence')


figure(3)
clf
imagesc(double(cat(1,behavior_pseudolabels{:})))
colormap(parula(numel(pseudolabel_cats)))
cb = colorbar;
ticks = linspace(cb.Limits(1), cb.Limits(2), numel(pseudolabel_cats)+1);
ticks = ticks(1:end-1) + mean(diff(ticks))/2;
cb.Ticks = ticks;
cb.TickLabels = pseudolabel_cats;
title('Pseudolabel occurrence')
ylabel('Trials')
xlabel('Video frames')