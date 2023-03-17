%% Text Klassifizierung

%% 
filenameTrain = 'factoryReports.csv';     % ships with DL toolbox
T = readtable(filenameTrain); 
load('textClassNet.mat'); 
%%
reportsNew = [ 
   "Coolant is pooling underneath sorter."
   "Sorter blows fuses at start up."
   "There are some very loud rattling sounds coming from the assembler."];

%% 
documents = lower(tokenizedDocument(reportsNew)); 
% Convert documents to embeddingDimension-by-sequenceLength-by-1 images.
emb = fastTextWordEmbedding;
predictors = doc2sequence(emb, documents, 'Length',14);

% Reshape data to be of size 1-by-sequenceLength-embeddingDimension.
predictors = cellfun(@(X) permute(X,[3 2 1]),predictors,'UniformOutput',false);
tbl = table;
tbl.Predictors = predictors;

%% 
labelsNew = classify(net,tbl)