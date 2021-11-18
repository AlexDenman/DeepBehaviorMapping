function [ microstate_labels, microstate_centroids ] = dbmExtractMicrostates( latent_trajectories, num_microstates, max_iter, replicates, use_parallel )
%%
arguments
    latent_trajectories cell
    num_microstates (1,1) {mustBeNumeric} = 50
    max_iter (1,1) {mustBeNumeric} = 1000
    replicates (1,1) {mustBeNumeric} = 100
    use_parallel (1,1) logical = false
end
    
[microstate_labels, microstate_centroids] = kmeans(cat(2,latent_trajectories{:}).', ...
    num_microstates, ...
    'MaxIter', max_iter, ...
    'Replicates', replicates, ...
    'Options', statset('UseParallel', use_parallel));

% Reformat microstate labels to have same shape as original data
microstate_labels = mat2cell(microstate_labels.', 1, cellfun(@(X) size(X,2), latent_trajectories));

%%
end