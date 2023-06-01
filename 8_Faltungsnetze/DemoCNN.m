%% Beispiele zu Faltungsnetzen

%% Bsp 1: MNIST-Ziffern, 6 Klassen

%% 1.1: Daten laden
load(fullfile('..', 'Datensaetze', 'mnist6Klassen.mat')); 

imSize = 28; 
nImgsTrain = length(trainLbl); 
nImgsTest = length(testLbl); 
C  = length(categories(trainLbl));  % Anzahl Klassen

% Umspeichern in 4D-Array mit 1 Farbkanal
trainImgs4D = reshape(trainImgs, imSize, imSize, 1, nImgsTrain);
testImgs4D = reshape(testImgs, imSize, imSize, 1, nImgsTest); 


%% 1.2: Zufällig ein paar Bilder auswählen und zeigen
idx = randperm(nImgsTrain, 20);
t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
title(t, 'Ein paar Trainingsbilder');
for i = 1:numel(idx)
    nexttile;
    imshow(trainImgs4D(:,:,1,idx(i)));
    title(sprintf('%s', trainLbl(idx(i))));
end

%% 1.3: Zum Vergleich: ein VV-Netz mit einer inneren Schicht
% Bei einer vollverbundenen Schicht nach dem Input-Layer ist es egal, ob man 
% die Bilder 'flach' als featureInputLayer oder in Matrixform als imageInputLayer 
% in das Netz schickt.

H = 100; 
layers = [imageInputLayer([imSize, imSize, 1])
   fullyConnectedLayer(H)
   reluLayer
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];

options = trainingOptions('rmsprop',...
   'MiniBatchSize', 128,...
   'MaxEpochs', 20, ...
   'Verbose', true, ...
   'ValidationData', {testImgs4D, testLbl},...
   'Plots', 'none');
% analyzeNetwork(layers);

% Training des Netzes und Berechnung der Fehler
[net, info] = trainNetwork(trainImgs4D, trainLbl, layers, options);
trainLblPred = classify(net, trainImgs4D); 
trainErr = mean(trainLbl ~= trainLblPred); 
testLblPred = classify(net, testImgs4D); 
testErr = mean(testLbl ~= testLblPred);
plotTrainingProgress(info);
%% 1.4: Ein Mini-Faltungsnetz

% 20 Faltungskerne der Größe 5x5
layers = [
    imageInputLayer([imSize, imSize, 1])
    convolution2dLayer(5,20, 'Stride',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(C)
    softmaxLayer
    classificationLayer];
analyzeNetwork(layers);

options = trainingOptions('rmsprop',...
   'MiniBatchSize', 128,...
   'MaxEpochs', 20, ...
   'Verbose', true, ...
   'ValidationData', {testImgs4D, testLbl},...
   'Plots', 'none');


% Training des Netzes und Berechnung der Fehler
[net, info] = trainNetwork(trainImgs4D, trainLbl, layers, options);
trainLblPred = classify(net, trainImgs4D); 
trainErr = mean(trainLbl ~= trainLblPred); 
testLblPred = classify(net, testImgs4D); 
testErr = mean(testLbl ~= testLblPred);
plotTrainingProgress(info, 20);

%% 1.5: Ein groesseres Faltungsnetz
% Aus ML-Examples
layers = [
   imageInputLayer([imSize imSize 1])
    
    convolution2dLayer(3,8,'Padding','same'); 
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)  % Padding=0
    
    convolution2dLayer(3,16,'Padding','same'); 
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same'); 
    batchNormalizationLayer
    reluLayer
       
    fullyConnectedLayer(C)
    softmaxLayer
    classificationLayer];
analyzeNetwork(layers);

options = trainingOptions('rmsprop',...
   'MiniBatchSize', 128,...
   'MaxEpochs', 20, ...
   'Verbose', true, ...
   'ValidationData', {testImgs4D, testLbl},...
   'Plots', 'none');

%% Training des Netzes und Berechnung der Fehler
[net, info] = trainNetwork(trainImgs4D, trainLbl, layers, options);
trainLblPred = classify(net, trainImgs4D); 
trainErr = mean(trainLbl ~= trainLblPred); 
testLblPred = classify(net, testImgs4D); 
testErr = mean(testLbl ~= testLblPred);
plotTrainingProgress(info, 20);

%% Bsp 2: Teil des Cifar10-Datensatzes
% https://www.cs.toronto.edu/~kriz/cifar.html

%% 2.1: Daten laden
imgSize = 32;   % [32 x 32 x 3] - Bilder
testDs = shuffle(imageDatastore(fullfile('..', 'Datensaetze', 'Cifar10Subset', 'test'), ...
   'IncludeSubfolders',true, 'LabelSource','foldernames'));
trainDs = shuffle(imageDatastore(fullfile('..', 'Datensaetze', 'Cifar10Subset', 'train'), ...
   'IncludeSubfolders',true, 'LabelSource','foldernames'));
tabulate(trainDs.Labels);
C = length(categories(trainDs.Labels));

%% 2.2 Ein paar Bilder ansehen

t = tiledlayout('flow'); % 4,5, 'TileSpacing','compact', 'Padding', 'compact');
for i = 1:20
   nexttile;
   [img, info] = trainDs.read(); 
   imshow(img); 
   title(sprintf('%s', info.Label)); 
end

%% 2.3: Eigenes kleines Netz definieren
layers = [
   imageInputLayer([imgSize, imgSize, 3])
   convolution2dLayer(3,12,'Padding','same')
   batchNormalizationLayer
   reluLayer
   maxPooling2dLayer(2,'Stride',2)
   
   convolution2dLayer(3,20,'Padding','same')
   batchNormalizationLayer
   reluLayer
   maxPooling2dLayer(2,'Stride',2)
   
   convolution2dLayer(3,40,'Padding','same')
   batchNormalizationLayer
   reluLayer
   
   dropoutLayer(0.5)
   
   fullyConnectedLayer(100)
   batchNormalizationLayer
   reluLayer
   
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];
% analyzeNetwork(layers);
%% 2.4: Trainineren und Fehler berechnen
options = trainingOptions('sgdm', ...
   'InitialLearnRate', 0.003, ...
   'MaxEpochs', 40, ...
   'MiniBatchSize',  128, ...
   'ValidationFrequency', 80, ...
   'ValidationData', testDs, ...
   'L2Regularization', 1/10, ...
   'Verbose',false, ...
   'Plots','none');
% Training
tic
[net, info] = trainNetwork(trainDs, layers, options);
toc
plotTrainingProgress(info); 
% Fehler
trainLblPred = classify(net, trainDs);
trainErr = mean(trainDs.Labels ~= trainLblPred); 

testLblPred = classify(net, testDs);
testErr = mean(testDs.Labels ~= testLblPred);

cmTest = confusionmat(testDs.Labels, testLblPred);
cc = confusionchart(cmTest, categories(testDs.Labels));
cc.ColumnSummary = 'column-normalized';
cc.RowSummary = 'row-normalized';
cc.Title = sprintf('Testfehler: %.1f%%', 100*testErr);

%% Bsp 3: Image Augmentation
% ds = imageDatastore("banane.jpg");
img = imread("banane.jpg"); 
aug = imageDataAugmenter('RandRotation', [0, 360], ...
   'RandScale', [0.5, 1.2], ...
   'RandXTranslation', [-50, 50], ...
   'RandYTranslation', [-50, 50], ...
   'FillValue', 70*[1,1,1], ...
   'RandXReflection', true, 'RandYReflection', true);

tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact');
nexttile; 
imshow(img);   % Originalbild
for i = 2:20
   nexttile;
   imshow(augment(aug, img)); 
end


%% 3.1: Anwenden auf Cifar-10
aug = imageDataAugmenter('RandXTranslation', [-3, 3], ...
   'RandYTranslation', [-3, 3], ...
   'RandXReflection', true);
augDs = augmentedImageDatastore([imgSize, imgSize], trainDs, ...
   'DataAugmentation', aug);
% Trainineren 
options.MaxEpochs = 60; 
tic
[net, info] = trainNetwork(augDs, layers, options);
toc
plotTrainingProgress(info); 
% Fehler
trainLblPred = classify(net, trainDs);
trainErr = mean(trainDs.Labels ~= trainLblPred); 
testLblPred = classify(net, testDs);
testErr = mean(testDs.Labels ~= testLblPred);

%% Bsp 4: Transfer-Learning

%% 4.1: googlenet laden
net = googlenet; 
inputSize = net.Layers(1).InputSize;
classNames = net.Layers(end).Classes; 

trainDsResized = augmentedImageDatastore(inputSize(1:2), trainDs); 
testDsResized = augmentedImageDatastore(inputSize(1:2), testDs);   

%% 4.2: Original-Googlenet auf die Daten anwenden
lbl = classify(net, trainDsResized);
tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact');
imgBatch= read(trainDsResized); 
for i = 1:20
   nexttile;
   img = imgBatch.input{i};
   [lbl, score] = classify(net, img); 
   imshow(img); 
   title({string(imgBatch.response(i)), sprintf('%s, %.f%%', string(lbl), 100* max(score))}); 
end

%% 4.3: Netz anpassen
lgraph = layerGraph(net); 
% die 3 hintersten Schichten entfernen
lgraph = removeLayers(lgraph, {'loss3-classifier', 'prob', 'output'});
% eigene Klassifizierungsschichten einbauen
lgraph = addLayers(lgraph, ...
   [fullyConnectedLayer(C, 'Name', 'myFC', ...
   'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
   softmaxLayer('Name', 'mySM')
   classificationLayer('Name', 'myClass')]);
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'myFC'); 


%% 4.4: vordere Schichten einfrieren (optional)
frozenLayers = lgraph.Layers; 
freezeIdx = 1:39; 
frozenLayers(freezeIdx) = freezeWeights(lgraph.Layers(freezeIdx));
lgraph =  createLgraphUsingConnections(frozenLayers, lgraph.Connections);

%% 4.5 Trainieren und Fehler berechnen
miniBatchSize = 64;
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',2, ...
    'LearnRateSchedule', 'piecewise', ...
    'InitialLearnRate',1e-3, ...
    'LearnRateDropPeriod', 1, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ValidationData',testDsResized, ...
    'ValidationFrequency', 10);
tic
netTransfer = trainNetwork(trainDsResized, ...
   lgraph,options);
toc
%
trainLblPred = classify(netTransfer, trainDsResized);
trainErr = mean(trainDs.Labels ~= trainLblPred); 
testLblPred = classify(netTransfer, testDsResized);
testErr = mean(testDs.Labels ~= testLblPred);

cmTest = confusionmat(testDs.Labels, testLblPred);
confusionchart(cmTest, categories(testDs.Labels), ...
   'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized', ...
   'Title', sprintf('Testfehler: %.1f%%', 100*testErr));
