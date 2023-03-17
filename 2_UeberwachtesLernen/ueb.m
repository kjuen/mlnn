% Uebungsaufgabe Kapitel 2

%% Teil 1
dateiName = fullfile('..', 'Datensaetze', 'binclass2D.csv');
T = readtable(dateiName); 
disp(head(T)); 
% Speichern als Datenmatrix und kategorielles Array
trainMat = [T.x, T.y]; 
trainLbl = categorical(T.Klasse);
tabulate(trainLbl); 
gscatter(trainMat(:,1), trainMat(:,2), trainLbl, 'rg','o'); 
title('Trainingsdaten'); 

%% kNN trainieren
k = 7; 
kNN = fitcknn(trainMat, trainLbl, 'NumNeighbors',k);


%% Testdatensatz einlesen und klassifizieren
dateiName = fullfile('..', 'Datensaetze', 'binclass2D_test.csv');
T = readtable(dateiName); 
testMat = [T.x, T.y]; 
testLblPred = predict(kNN, testMat);
gscatter(testMat(:,1), testMat(:,2), testLblPred, 'rg','o'); 
title('Testdaten'); 

%% Klassifizerungsregionen
nGrid = 100; 
[X,Y] = meshgrid(linspace(-4, 4, nGrid));
[~, S] = predict(kNN, [X(:), Y(:)]);
img = zeros(nGrid, nGrid, 3);
img(:,:,1) = reshape(S(:,1), nGrid, nGrid);
img(:,:,2) = reshape(S(:,2), nGrid, nGrid);
hold on 
image(img, 'XData', [-4,4], 'YData', [-4,4]);
alpha(0.4);   % Transparenzstufe
hold off; 
axis([-4, 4, -4, 4]);

%% Teil 2
trainLblPred = predict(kNN, trainMat);
trainErr = sum(trainLblPred ~= trainLbl) / length(trainLbl);
testLbl = categorical(T.Klasse);
testLblPred = predict(kNN, testMat);
testErr = sum(testLblPred ~= testLbl) / length(testLbl);

%% Mit Normierung (bringt hier nichts) 
m = mean(trainMat);
s = std(trainMat);
trainMatNorm = (trainMat - m) ./ s;
testMatNorm = (testMat - m) ./ s;

k = 7; 
kNN = fitcknn(trainMatNorm, trainLbl, 'NumNeighbors',k);
trainLblPred = predict(kNN, trainMatNorm);
trainErrNorm = sum(trainLblPred ~= trainLbl) / length(trainLbl);
testLblPred = predict(kNN, testMatNorm);
testErrNorm = sum(testLblPred ~= testLbl) / length(testLbl);

