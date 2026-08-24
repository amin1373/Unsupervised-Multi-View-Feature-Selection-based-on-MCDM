function [Y] = LaplacianScore(X, W)
%         W = dist(X_train');
%         W = -W./max(max(W)); % it's a similarity
%         [lscores] = LaplacianScore(X_train, W);
%         [junk, ranking] = sort(-lscores);
%	Usage:
%	[Y] = LaplacianScore(X, W)
%
%	X: Rows of vectors of data points
%	W: The affinity matrix.
%	Y: Vector of (1-LaplacianScore) for each feature.
%      The features with larger y are more important.
%
%    Examples:
%
%       fea = rand(50,70);
%       options = [];
%       options.Metric = 'Cosine';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'Cosine';
%       W = constructW(fea,options);
% 
%       LaplacianScore = LaplacianScore(fea,W);
%       [junk, index] = sort(-LaplacianScore);
%       
%       newfea = fea(:,index);
%       %the features in newfea will be sorted based on their importance.
%
%	Type "LaplacianScore" for a self-demo.
%
% See also constructW
%
%Reference:
%
%   Xiaofei He, Deng Cai and Partha Niyogi, "Laplacian Score for Feature Selection".
%   Advances in Neural Information Processing Systems 18 (NIPS 2005),
%   Vancouver, Canada, 2005.   
%
% %   Deng Cai, 2004/08
% @INPROCEEDINGS{HCN05a,
%          AUTHOR = {Xiaofei He and Deng Cai and Partha Niyogi},
%          TITLE = {Laplacian Score for Feature Selection},
%          BOOKTITLE = {Advances in Neural Information Processing Systems 18},
%          YEAR = {2005}, }
% fprintf('\n+ Feature selection method: Laplacian Score \n');
%  Version 5.0 August 2017
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk
%  If you use our toolbox please cite our supporting papers:
% 
%  BibTex
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV17, 
% author={Giorgio Roffo and Simone Melzi and Umberto Castellani and Alessandro Vinciarelli}, 
% booktitle={2017 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach}, 
% year={2017}, 
% month={Oct}}
%  ------------------------------------------------------------------------
% @Inbook{Roffo2017,
% author="Roffo, Giorgio
% and Melzi, Simone",
% editor="Appice, Annalisa
% and Ceci, Michelangelo
% and Loglisci, Corrado
% and Masciari, Elio
% and Ra{\'{s}}, Zbigniew W.",
% title="Ranking to Learn:",
% bookTitle="New Frontiers in Mining Complex Patterns: 5th International Workshop, NFMCP 2016, Held in Conjunction with ECML-PKDD 2016, Riva del Garda, Italy, September 19, 2016, Revised Selected Papers",
% year="2017",
% publisher="Springer International Publishing",
% address="Cham",
% pages="19--35",
% isbn="978-3-319-61461-8",
% doi="10.1007/978-3-319-61461-8_2",
% url="https://doi.org/10.1007/978-3-319-61461-8_2"
% }
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV15, 
% author={G. Roffo and S. Melzi and M. Cristani}, 
% booktitle={2015 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Feature Selection}, 
% year={2015}, 
% pages={4202-4210}, 
% doi={10.1109/ICCV.2015.478}, 
% month={Dec}}
%  ------------------------------------------------------------------------
if nargin == 0, selfdemo; return; end
[nSmp,nFea] = size(X);
if size(W,1) ~= nSmp
    error('W is error');
end
D = full(sum(W,2));
L = W;
allone = ones(nSmp,1);
tmp1 = D'*X;
D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);
DPrime = sum((X'*D)'.*X)-tmp1.*tmp1/sum(diag(D));
LPrime = sum((X'*L)'.*X)-tmp1.*tmp1/sum(diag(D));
DPrime(find(DPrime < 1e-12)) = 10000;
Y = LPrime./DPrime;
Y = Y';
Y = full(Y);
    
