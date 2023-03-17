%% kNN als Beispiel fuer einen Lernalgo

%% Trainingsdaten laden
dateiName = fullfile('..', 'Datensaetze', 'hundkatzemaus.csv');
T = readtable(dateiName); 
disp(head(T)); 
% Speichern als Datenmatrix und kategorielles Array
trainMat = [T.Pfotengroesse, T.Kopflaenge]; 
trainLbl = categorical(T.Tierart);
tabulate(trainLbl); 

%% Scatterplot
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact'); 
nexttile; 
gscatter(trainMat(:,1), trainMat(:,2), trainLbl, 'rgb','o'); 
title('Trainingsdaten'); 
xlabel(T.Properties.VariableNames{1}), ylabel(T.Properties.VariableNames{2}); 
xdim = [-4,4]; 
ydim = [-8, 6]; 
axis([xdim, ydim]); 

%% Trainieren des knn

k = 5; 
kNN = fitcknn(trainMat, trainLbl, 'NumNeighbors',k);

%% Anwenden auf die Daten, die wir haben:
trainLblPred = predict(kNN, trainMat); 
nexttile; 
gscatter(trainMat(:,1), trainMat(:,2), trainLblPred, 'rgb','o'); 
axis([xdim, ydim]);
title('Trainingsdaten mit kNN-Labeln'); 
mask = trainLblPred ~= trainLbl; 

%% Fehlklassifizierungen sichtbar machen
hold on; 
scatter(trainMat(mask,1), trainMat(mask, 2), 72, 'kx', 'Linewidth', 2, ...
   'DisplayName', 'Trainingsfehler');
legend off;
hold off 

%% Klassifizierungsregionen
nGrid = 200; 
[X,Y] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
    linspace(ydim(1), ydim(2), nGrid)); 
[~, S] = predict(kNN, [X(:), Y(:)]);
S = reshape(S, [nGrid, nGrid, 3]);
hold on; 
image(S, 'XData', xdim, 'YData', ydim);
alpha(0.4);   % Transparenzstufe
hold off; 
axis([xdim, ydim]);





%% Berechnung Trainingsfehler
trainErr = sum(trainLblPred ~= trainLbl) / length(trainLblPred);
trainAcc = 1 - trainErr; 



%% Testen mit weiterem Datensatz 
dateiName = fullfile('..', 'Datensaetze', 'hundkatzemaus_test.csv');
Ttest = readtable(dateiName); 
disp(head(Ttest)); 
testMat = [Ttest.Pfotengroesse, Ttest.Kopflaenge]; 
testLbl = categorical(Ttest.Tierart);
tabulate(testLbl); 

%% Scatterplot
figure; 
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact'); 
nexttile; 
gscatter(testMat(:,1), testMat(:,2), testLbl, 'rgb', 'o'); 
title('Testdaten'); 
axis([xdim, ydim]); 

%% Anwenden von kNN
testLblPred = predict(kNN, testMat); 
nexttile; 
gscatter(testMat(:,1), testMat(:,2), testLblPred, 'rgb', 'o'); 
title('Testdaten mit kNN Labeln'); 
axis([xdim, ydim]); 

%% Analyse der Testfehler
testErr = sum(testLblPred ~= testLbl) / length(testLbl);
testAcc = 1 - testErr; 
mask = testLbl ~= testLblPred;
hold on; 
scatter(testMat(mask, 1), testMat(mask, 2), 72, 'kx', 'Linewidth', 2, ...
   'DisplayName', 'Testfehler');
hold off 
title(sprintf('Testdaten mit kNN Labeln, Testfehler=%.2f%%', testErr*100)); 
legend off; 

%% Kreuzvalidierung mit den Trainingsdaten
nChunks = 5;   % mit 80% wird jeweils trainiert
nRuns = 50; 
trainFunc = @(dat,lbl) fitcknn(dat,lbl, 'NumNeighbors',k);
tic
[xTestErr, xTrainErr] = xval(trainFunc, @predict, ...
    trainMat, trainLbl, nChunks, nRuns);
toc
 
%% Darstellung als Histogramm
figure;
histogram(xTestErr); 
hold on; histogram(xTrainErr); hold off; 
legend('Testfehler', 'Trainingsfehler'); 
title({sprintf('kNN, k= %i', k); 
    sprintf('Mittlerer Testfehler: %.3f%%', 100*mean(xTestErr)); 
    sprintf('Mittlerer Trainingsfehler: %.3f%%', 100*mean(xTrainErr));});

%% Besten Wert fuer k suchen
nChunks = 5;   % mit 80% wird jeweils trainiert
nRuns = 20;
kVec = [1:21, 23:2:41, 45:5:90];
nk = length(kVec);
meanTestErr = zeros(nk,1); 
meanTrainErr = zeros(nk,1); 
tic;
for kk=1:nk
    disp(kVec(kk))
    trainFunc = @(dat,lbl) fitcknn(dat,lbl, 'NumNeighbors', kVec(kk));
    [xTestErr, xTrainErr] = xval(trainFunc, @predict, ...
       trainMat, trainLbl, nChunks, nRuns);
    meanTestErr(kk) = mean(xTestErr); 
    meanTrainErr(kk) = mean(xTrainErr); 
end
toc

%% Graphisch darstellen
plot(kVec, meanTestErr, 'r', kVec, meanTrainErr, 'b', 'Linewidth', 2); 
legend('Testfehler', 'Trainingsfehler'); 
ylim([0, 0.3]);
xlabel('Anzahl Nachbarn k'); 
ylabel('Fehlerrate'); 
title('Kreuzvalidierung knn'); 

 
%% knn und Spiral data
load(fullfile('..', 'Datensaetze', 'spiral')); 
xdim = [min(xySpiral(:,1)), max(xySpiral(:,1))]; 
ydim = [min(xySpiral(:,2)), max(xySpiral(:,2))];
gscatter(xySpiral(:,1), xySpiral(:,2), lblSpiral, 'rg', '..', 15*[1,1]);
xlim(xdim), ylim(ydim); 

%% knn anwenden
k =250;
kNN = fitcknn(xySpiral, lblSpiral, 'NumNeighbors',k); 
% Klassifikationsregionen sichtbar machen
nGrid = 250; 
[X,Y] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
    linspace(xdim(1), xdim(2), nGrid)); 
Z = predict(kNN, [X(:), Y(:)]);
Z = reshape(Z, size(X)); 
Z = cat(3, 1-Z, Z, zeros(size(X)));   % b-Kanal = 0
hold on; 
image(Z, 'XData', xdim, 'YData', ydim);
alpha(0.4);   % Transparenzstufe
hold off; 
axis([xdim, ydim]);

%% Kreuzvalidierung
nChunks = 5;   % mit 80% wird jeweils trainiert
nRuns = 5;
kVec = [1:21, 23:2:41, 45:5:175];
nk = length(kVec);
meanTestErr = zeros(nk,1); 
meanTrainErr = zeros(nk,1); 
tic;
for kk=1:nk
    disp(kVec(kk))
    trainFunc = @(dat,lbl) fitcknn(dat,lbl, 'NumNeighbors', kVec(kk));
    [xTestErr, xTrainErr] = xval(trainFunc, @predict, ...
       xySpiral, lblSpiral, nChunks, nRuns);
    meanTestErr(kk) = mean(xTestErr); 
    meanTrainErr(kk) = mean(xTrainErr); 
end
toc

%% Graphisch darstellen
plot(kVec, meanTestErr, 'r', kVec, meanTrainErr, 'b', 'Linewidth', 2); 
legend('Testfehler', 'Trainingsfehler'); 
ylim([0, 0.3]);
xlabel('Anzahl Nachbarn k'); 
ylabel('Fehlerrate'); 
title('Kreuzvalidierung knn'); 

