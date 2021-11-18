function [ pose_sequences ] = dbmFormatInputs( body_parts_X, body_parts_Y )
%%


%% Distances between body parts within each frame
bodypart_distances = cell(size(body_parts_X{1},1)-1,1);
for n = 1:numel(bodypart_distances)
    x_dists = cellfun(@(X) X(n,:) - X((n+1):end,:), body_parts_X, 'UniformOutput', false);
    y_dists = cellfun(@(Y) Y(n,:) - Y((n+1):end,:), body_parts_Y, 'UniformOutput', false);
    bodypart_distances{n} = cellfun(@(X,Y) sqrt(X.^2 + Y.^2), x_dists, y_dists, 'UniformOutput', false);
end

bodypart_distances = num2cell(cat(2,bodypart_distances{:}),2);
bodypart_distances = cellfun(@(X) cat(1,X{:}), bodypart_distances, 'UniformOutput', false);
%% Distance between each body part in current frame and all body parts in next frame

frame_to_frame_distances = cell(size(body_parts_X{1},1),1);
for n = 1:numel(frame_to_frame_distances)
    x_dists = cellfun(@(X) X(n,2:end) - X(:,1:end-1), body_parts_X, 'UniformOutput', false);
    y_dists = cellfun(@(Y) Y(n,2:end) - Y(:,1:end-1), body_parts_Y, 'UniformOutput', false);
    frame_to_frame_distances{n} = cellfun(@(X,Y) sqrt(X.^2 + Y.^2), x_dists, y_dists, 'UniformOutput', false);
end

frame_to_frame_distances = num2cell(cat(2,frame_to_frame_distances{:}),2);
frame_to_frame_distances = cellfun(@(X) cat(1,X{:}), frame_to_frame_distances, 'UniformOutput', false);

% Pad sequences by repeating last value
frame_to_frame_distances = cellfun(@(X) [X X(:,end)], frame_to_frame_distances, 'UniformOutput', false);


%% Join 
pose_sequences = cellfun(@vertcat, bodypart_distances, frame_to_frame_distances, 'UniformOutput', false);


%%
end