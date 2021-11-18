function [ pseudolabel_predictions, latent_trajectories ] = dbmGetModelOutputs( dbm_mdl, pose_sequences, use_gpu )
%%

arguments
    dbm_mdl 
    pose_sequences cell
    use_gpu (1,1) logical = false
end

if use_gpu
    exec_env = 'gpu';
else
    exec_env = 'cpu';
end

pseudolabel_predictions = cellfun(@(X) predict(dbm_mdl, X, 'ExecutionEnvironment', exec_env), pose_sequences, 'UniformOutput', false);
latent_trajectories = cellfun(@(X) cell2mat(activations(dbm_mdl, X, 'lstm1', 'ExecutionEnvironment', exec_env)), pose_sequences, 'UniformOutput', false);


%%
end