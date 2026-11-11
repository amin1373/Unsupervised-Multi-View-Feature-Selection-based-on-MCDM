clc;clear all;

addpath('./FL');
addpath('./gspbox-master');
addpath('./MinMaxSelection');
addpath('./util');
addpath('./util1');
addpath('./util2');
addpath('./utils');
addpath('./src');
addpath('./datasets');
gsp_start;
a='yale_mtv';

filename=['datasets\',a,'.mat'];
load(filename);

[~,v]=size(fea);
class_num=size(unique(gt),1);
label=gt;
allP=[];
X1=[];
numViews = length(fea); 
normalizedFea = cell(1, numViews); 

for i = 1:numViews
    currentViewData = fea{i};
    
    normalizedFea{i} = minMaxNormalize(currentViewData);
end
fea=normalizedFea;

%%
n=size(fea{1},1);
for num = 1:v
    X1=[X1,fea{num}];
end
[J1,J2]=size(X1);
feanum=[10:10:100]; 


%% MVFS-MCDM
start=tic;
Decision_Matrix= [];
numViews = v;

viewSpecificData = cell(1, numViews); 
concatenatedData = cell(1, numViews); 

for i = 1:numViews
    currentViewData = fea{i};

    viewSpecificData{i} = currentViewData;

    otherViewsData = [];
    for j = 1:numViews
        if j ~= i
            otherViewsData = [otherViewsData, fea{j}];
        end
    end

    [~,feature_num]=size(currentViewData);
    % Intra-view redundancy   
    H1 = pdist2(currentViewData',currentViewData','cosine'); 
    H1=nanmean(H1,2);

    % Inter-view redundancy   
    H2 = pdist2(currentViewData',otherViewsData','cosine'); 
    H2=nanmean(H2,2);



        %Laplacian Score       
        % M6=zeros(feature_num,1);
        % W = dist(currentViewData');
        % W = -W./max(max(W)); % it's a similarity
        % [H3] = LaplacianScore(currentViewData, W);
        [H3,h3Info{i}] = compute_H3_knn_safe(X,graphK);

    H = [H1,H2,H3];
    Decision_Matrix= [Decision_Matrix;H];
end

% Normalize Decision_Matrix using Min-Max normalization
for col = 1:size(Decision_Matrix, 2)
    minValue = min(Decision_Matrix(:, col));
    maxValue = max(Decision_Matrix(:, col));

    % Avoid division by zero for constant columns
    if maxValue - minValue == 0
        Decision_Matrix(:, col) = 0; % Assign zero if all values are the same
    else
        Decision_Matrix(:, col) = (Decision_Matrix(:, col) - minValue) / (maxValue - minValue);
    end
end

metricVariance = nanvar(Decision_Matrix, 0, 1); 
W = metricVariance / sum(metricVariance); 


[fr,fc]=size(Decision_Matrix);
 E5=Fun_TOPSIS(fr,fc,Decision_Matrix,W,ones(1,fc),inf, 4);

[~,S1] = sort(E5,'descend');
index=S1;
MVFS2_time = toc(start);

ACC=[];
NMI=[];
ACCstd=[];
NMIstd=[];
Fscore=[];
Fscorestd=[];
Precision=[];
Precisionstd=[];
Recall=[];
Recallstd=[];
ARI=[];
ARIstd=[];

for j = 1:length(feanum)
acc=[];
nmi=[];
fscore=[];
precision=[];
recall=[];
ari=[];
for k = 1:20
    new_fea = X1(:,index(1:feanum(j)));
    idx = kmeans(new_fea, class_num,'MaxIter',1000);
    res = bestMap(label,idx);
    acc111 = length(find(label == res))/length(label); % calculate ACC 
    nmi111 = MutualInfo(label,idx); % calculate NMI
    [f111,p111,r111] = compute_f(label,idx);
%    [ar111,~,~,~]=RandIndex(label,idx);
    acc=[acc;acc111];
    nmi=[nmi;nmi111];
    fscore=[fscore;f111];
    precision=[precision;p111];
    recall=[recall;r111];
   % ari=[ari;ar111];
end
ACC=[ACC;sum(acc)/20];
ACCstd=[ACCstd;std(acc)];
NMI=[NMI;sum(nmi)/20];
NMIstd=[NMIstd;std(nmi)];
Fscore=[Fscore;sum(fscore)/20];
Fscorestd=[Fscorestd;std(fscore)];
Precision=[Precision;sum(precision)/20];
Precisionstd=[Precisionstd;std(precision)];
Recal1=[Recall;sum(recall)/20];
Recallstd=[Recallstd;std(recall)];
%ARI=[ARI;sum(ari)/20];
%ARIstd=[ARIstd;std(ari)];
end

MVFS2_ACC=ACC;
MVFS2_ACCstd=ACCstd;
MVFS2_NMI=NMI;
MVFS2_NMIstd=NMIstd;
MVFS2_Fscore=Fscore;
MVFS2_Fscorestd=Fscorestd;
MVFS2_Precision=Precision;
MVFS2_Precisionstd=Precisionstd;
MVFS2_Recall=Recall;
MVFS2_Recallstd=Recallstd;
MVFS2_ARI=ARI;
MVFS2_ARIstd=ARIstd;
