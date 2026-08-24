function normalizedData = minMaxNormalize(data)
    % Compute min and max values for each feature (column-wise)
    minValues = min(data, [], 1);
    maxValues = max(data, [], 1);
    
    % Apply Min-Max normalization
    normalizedData = (data - minValues) ./ (maxValues - minValues);
    
    % Handle constant features (where max = min)
    normalizedData(:, maxValues == minValues) = 0; % Set to 0 for constant features
end