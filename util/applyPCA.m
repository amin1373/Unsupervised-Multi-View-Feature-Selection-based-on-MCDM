function reducedFea = applyPCA(fea, varianceRetained)
    % Apply PCA to reduce dimensionality while retaining specified variance
    % Supports both dense and sparse matrices, and n < d cases
    % fea: cell array of views (can be full or sparse)
    % varianceRetained: percentage of variance to retain (e.g., 0.95)
    
    numViews = length(fea);
    reducedFea = cell(1, numViews);
    
    for v = 1:numViews
        currentView = fea{v};
        
        % Convert sparse to full if necessary (for small to medium datasets)
        if issparse(currentView)
            % Check if converting to full is memory-safe
            [n, d] = size(currentView);
            if n * d < 1e7 % If less than 10 million elements, convert to full
                currentView = full(currentView);
            else
                % For very large sparse matrices, use specialized approach
                fprintf('View %d: Large sparse matrix (%d × %d). Using svds...\n', v, n, d);
                reducedFea{v} = applyPCAToSparse(currentView, varianceRetained);
                continue;
            end
        end
        
        [n, d] = size(currentView);
        
        % Center the data
        meanVal = mean(currentView, 1);
        centeredData = currentView - meanVal;
        
        % Determine maximum number of components
        maxComponents = min(n, d);
        
        % Use MATLAB's built-in pca (handles n < d and sparse properly)
        % Note: pca automatically converts sparse to full
        try
            [~, score, latent] = pca(centeredData);
        catch
            % Fallback: if pca fails, use svd approach
            fprintf('View %d: pca failed. Using fallback method...\n', v);
            [U, S, ~] = svd(centeredData, 'econ');
            score = U * S;
            latent = diag(S).^2 / (n - 1);
        end
        
        % Determine number of components to retain varianceRetained
        if sum(latent) > 0
            explainedVariance = latent / sum(latent);
            cumVariance = cumsum(explainedVariance);
            nComponents = find(cumVariance >= varianceRetained, 1, 'first');
        else
            nComponents = maxComponents;
        end
        
        if isempty(nComponents) || nComponents < 1
            nComponents = min(maxComponents, 100); % Default to 100 components
        end
        
        % Ensure we don't exceed available components
        nComponents = min(nComponents, size(score, 2));
        
        reducedFea{v} = score(:, 1:nComponents);
        
        %fprintf('View %d: %d → %d features (%.1f%% reduction)\n', ...
               % v, d, nComponents, (1 - nComponents/d) * 100);
    end
end

function reducedData = applyPCAToSparse(sparseData, varianceRetained)
    % Specialized PCA for large sparse matrices using svds
    [n, d] = size(sparseData);
    
    % Center the sparse data (mean is computed but not explicitly subtracted
    % from the sparse matrix to avoid densification)
    meanVal = mean(sparseData, 1);
    
    % Use svds on the centered data using a function handle
    % This avoids creating a dense centered matrix
    centerFun = @(x) x - meanVal' * (ones(1, size(x, 2)));
    
    % Number of components to compute (use min(n, d) but cap at 500)
    k = min(min(n, d), 500);
    
    try
        [U, S, ~] = svds(@(x) centerFun(centerFun(x')'), [n, d], k);
        score = U * S;
        latent = diag(S).^2 / (n - 1);
    catch
        % Fallback: use randomized PCA
        fprintf('    Using randomized PCA for sparse data...\n');
        [U, S, ~] = svds(sparseData, min(k, 100));
        score = U * S;
        latent = diag(S).^2 / (n - 1);
    end
    
    % Determine components
    if sum(latent) > 0
        explainedVariance = latent / sum(latent);
        cumVariance = cumsum(explainedVariance);
        nComponents = find(cumVariance >= varianceRetained, 1, 'first');
    else
        nComponents = min(k, 100);
    end
    
    if isempty(nComponents) || nComponents < 1
        nComponents = min(k, 100);
    end
    
    nComponents = min(nComponents, size(score, 2));
    reducedData = score(:, 1:nComponents);
end