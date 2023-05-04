%% Softmax-Klassifizierung
addpath(fullfile('..', '5_Gradientenabstieg'));  % fuer ga-Funktionen

%% Bsp 1: 2 Gruppen in 1D
load(fullfile('..', 'Datensaetze', 'Data1D2Gr_1.mat')); 
trainLbl = categorical(trainLbl);

fD = figure('WindowStyle', 'docked'); 
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact'); 
nexttile; 
gscatter(xTrain, double(trainLbl)-1, trainLbl, 'rg','ox'); 
ylim([-0.1, 1.1]); 
xlabel('x'); 
legend; 
nexttile;
histogram(xTrain(trainLbl=="1"), 20, 'FaceColor', 'r');
hold on;
histogram(xTrain(trainLbl=="2"), 20, 'FaceColor', 'g')
hold off;
title('Histogramm der Daten');


%% 2.1) Groessen-Parameter
N = length(trainLbl);
P = 1;    % 1 Merkmal
C = 2;    % 2 Klassen

%% 2.2) Anwenden eines vortrainierten Softmax-Klassifikators
wMat = [-1.11, 1.11];    % PxC
bVec = [1.72, -1.72];    % 1xC

[trainLblPred, pMatTrain] = applySoftmaxKlass(xTrain, wMat, bVec); 
errTrain = sum(trainLblPred ~= trainLbl) / length(trainLbl); 

%% 2.3) Einzeichnen der Wahrscheinlichkeiten
figure(fD); 
xx = linspace(-4,4)';   % Diskretisierung der der x-Achse
[~, ppMat] = applySoftmaxKlass(xx, wMat, bVec);   
nexttile(1);   % wieder in oberen Plot zeichnen
hold on; 
plot(xx, ppMat(:,1), 'r', 'DisplayName', 'P_1(x)');
plot(xx, ppMat(:,2), 'g', 'DisplayName', 'P_2(x)');
hold off; 
ylabel('Prob'); 
title('Daten und Wahrscheinlichkeiten'); 
xlim([-4,4]); 
legend('Location', 'west');


%% 2.4) Anwenden des Gradientenabstiegs


% Kreuzentropie als Kostenfunktion
[xs, dxs] = xsSoftmaxKlass(xTrain, trainLbl); 
% Anfangswerte
w0 = zeros(P,C); 
b0 = zeros(1,C); 
% Fuer den Gradientenabstieg: alle Parameter als ein Spaltenvektor!
wbVec0 = [w0(:); b0']; 
eta = 0.01; 
nIts = 100; 
[wbopt, track] = gaEinfach(dxs, eta, wbVec0, nIts);
% Umwandeln der Parameter in Gewichtsmatrix und Offset
wMat = reshape(wbopt(1:(P*C)), P, C);
bVec = reshape(wbopt((C*P+1):end), 1, C);

%% 2.5) Verlauf des Gradientenabstiegs

% Berechnung der Kostenfunktion entlang des Tracks
xsVec = zeros(nIts,1);
for n = 1:nIts
   xsVec(n) = xs(track(:,n));
end
nexttile(2); 
plot(1:nIts, xsVec);
xlabel('Iterationen'), ylabel('XS(w,b)'); 
title('Verlauf der Kostenfunktion')


%% Bsp 2: Hund-Katze-Maus-Datensatz (2D mit 3 Klassen)

% Trainingsdaten laden und plotten
dateiName = fullfile('..', 'Datensaetze', 'hundkatzemaus.csv');
T = readtable(dateiName); 
disp(head(T)); 
% Speichern als Datenmatrix und kategorielles Array
trainMat = [T.Pfotengroesse, T.Kopflaenge]; 
trainLbl = categorical(T.Tierart);
tabulate(trainLbl); 
% Daten plotten: 
f1 = figure;
gscatter(trainMat(:,1), trainMat(:,2), trainLbl, 'rgb','o'); 
title('Trainingsdaten'); 
xlabel(T.Properties.VariableNames{1}), ylabel(T.Properties.VariableNames{2}); 
xdim = [-4,4]; 
ydim = [-8, 6]; 
axis([xdim, ydim]); 

%% 2.1) Groessen-Parameter
N = length(trainLbl);
P = 2;    % 2 Merkmale
C = 3;    % 3 Klassen

%% 2.2) Anwenden des Gradientenabstiegs
% Kreuzentropie als Kostenfunktion
[xs, dxs] = xsSoftmaxKlass(trainMat, trainLbl); 
% Anfangswerte
w0 = zeros(P,C); 
b0 = zeros(1,C); 
% Fuer den Gradientenabstieg: alle Parameter als ein Spaltenvektor!
wbVec0 = [w0(:); b0']; 
eta = 0.01; 
nIts = 100; 
[wbopt, track] = gaEinfach(dxs, eta, wbVec0, nIts);
% Umwandeln der Parameter in Gewichtsmatrix und Offset
wMat = reshape(wbopt(1:(P*C)), P, C);
bVec = reshape(wbopt((C*P+1):end), 1, C);

[trainLblPred, pMatTrain] = applySoftmaxKlass(trainMat, wMat, bVec); 
% Angleichen der Klassennamen bei den vorhergesagten Labeln
trainLblPred = categorical(trainLblPred, categories(trainLblPred), categories(trainLbl));
% Berechnen der Fehlerrate
errTrain = sum(trainLblPred ~= trainLbl) / length(trainLbl); 

%% 2.3) Klassifizierungsregionen sichtbar machen
nGrid = 250; 
[X,Y] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
    linspace(ydim(1), ydim(2), nGrid)); 
[~, pMatGrid] = applySoftmaxKlass([X(:), Y(:)], wMat, bVec);
img = reshape(pMatGrid(:), nGrid, nGrid, 3); 
figure(f1);
hold on; 
image(img, 'XData', xdim, 'YData', ydim);
alpha(0.4);   % Transparenzstufe
hold off; 

%% 2.4) Die Wahrscheinlichkeiten als Surface-Plot darstellen
% Code von Kap. 3 (Bayes) kopiert
f3 = figure;
Cr = zeros(size(img)); 
Cr(:,:,1) = 1; 
surf(X,Y,img(:,:,1), Cr); 
shading interp; 
alpha(0.5)
hold on; 
Cg = zeros(size(img)); 
Cg(:,:,2) = 1; 
surf(X,Y,img(:,:,2), Cg); 
shading interp; 
alpha(0.5)

Cb = zeros(size(img)); 
Cb(:,:,3) = 1; 
surf(X,Y,img(:,:,3), Cb); 
shading interp; 
alpha(0.5)
hold off; 

%% Bsp 3: MNIST-Ziffern, 6 Klassen mit Softmax-Regression

load(fullfile('..', 'Datensaetze', 'mnist6Klassen.mat')); 

imSize = 28; 
nImgsTrain = length(trainLbl); 
nImgsTest = length(testLbl); 

%% 3.1) Zufällig ein paar Bilder auswählen und zeigen
idx = randperm(nImgsTrain, 20);
t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
title(t, 'Ein paar Trainingsbilder');
for i = 1:numel(idx)
    nexttile;
    imshow(trainImgs(:,:,idx(i)));
    title(sprintf('%s', trainLbl(idx(i))));
end

%% 3.2) Datenmatrizen erstellen
% Ein Bild ist eine Zeile. Dabei werden die Pixel alle
% in einer Zeile hintereinandergehängt
trainMat = zeros(nImgsTrain, imSize*imSize); 
for ii = 1:nImgsTrain
    img = trainImgs(:,:,ii);
    trainMat(ii, :) = img(:);
end
testMat = zeros(nImgsTest, imSize*imSize); 
for ii = 1:nImgsTest
    img = testImgs(:,:,ii);
    testMat(ii, :) = img(:);
end
P = size(trainMat,2);    % Anzahl Parameter = Anz Komp des w-Vektors
C = length(categories(trainLbl));   % Anzahl Klassen


%% 3.3) Deterministischer Gradientenabstieg
lambda = 10000;
[xs, dxs] = xsSoftmaxKlass(trainMat, trainLbl, lambda); 
w0 = 0.001 * randn(P,C);
b0 = 0.0 * rand(1, C);
wbVec0 = [w0(:); b0'];
nIts = 500;
eta = 0.005; 
 
tic
[wbopt, track] = gaRmsProp(dxs, eta, wbVec0, nIts, 0.95); 
toc

wMat = reshape(wbopt(1:(P*C)), P, C);
bVec = reshape(wbopt((C*P+1):end), 1, C);
%% 3.3.1) Trainings- und Testfehler
trainLblPred = applySoftmaxKlass(trainMat, wMat, bVec); 
% Angleichen der Klassenlabel
trainLblPred = categorical(trainLblPred, categories(trainLblPred), categories(trainLbl));
errTrain = sum(trainLblPred ~= trainLbl) / length(trainLbl);

testLblPred = applySoftmaxKlass(testMat, wMat, bVec); 
testLblPred = categorical(testLblPred, categories(testLblPred), categories(testLbl));
errTest = sum(testLblPred ~= testLbl) / length(testLbl);

%% 3.3.2) Confusion-Matrix
cmTest = confusionmat(testLbl, testLblPred);

cc = confusionchart(cmTest, categories(testLbl));
cc.ColumnSummary = 'column-normalized'; 
cc.RowSummary = 'row-normalized';
cc.Title = sprintf('MNIST (0-5) Confusion Matrix, Err-Rate=%.1f%%', 100*errTest);


%% 3.3.3) Verlauf der Kostenfunktion
xsVec = zeros(nIts,1);
normGradVec = zeros(nIts, 1);
for n = 1:nIts
   xsVec(n) = xs(track(:,n));
   normGradVec(n) = norm(dxs(track(:,n)));
end
figure('WindowStyle', 'docked'); 
yyaxis left; 
plot(1:nIts, xsVec);
ylabel('XS(w,b)'); 
yyaxis right; 
plot(1:nIts, normGradVec);
ylabel('| \nabla XS(w,b)|'); 
xlabel('Iterationen'); 
title('Verlauf der Kostenfunktion (det. GA)')
% ylim([0, 2*xsVec(end)]);


%% 3.3.4) Ein paar Testbilder mit vorhergesagten Labels zeigen

% idxFehl = find(testLblPred ~= testLbl);
% idx = idxFehl(randperm(length(idxFehl), 20));
% idx = randperm(nImgsTest, 20);
t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
title(t, 'Ein paar Testbilder');
for i = 1:numel(idx)
    nexttile;
    imshow(testImgs(:,:,idx(i)));
    title(sprintf('%s (vorh. %s)', testLbl(idx(i)), testLblPred(idx(i))));
end

%% 3.4) Stochastischer Gradientenabstieg
lambda = 0.001; 
[xs, dxs, dxsBatch] = xsSoftmaxKlass(trainMat, trainLbl, lambda); 
nEpochs = 30;
mbSize = 150;  
eta = 0.2;
w0 = 0.001 * randn(P,C);
b0 = 0.0 * rand(1, C);
wbVec0 = [w0(:); b0'];
tic
[wbopt, track] = sgaEinfach(dxsBatch, eta, wbVec0, ...
   nImgsTrain, nEpochs, mbSize);
toc
wMat = reshape(wbopt(1:(P*C)), P, C);
bVec = reshape(wbopt((C*P+1):end), 1, C);

%% 3.4.1) Trainings- und Testfehler
[trainLblPred, pMatTrain] = applySoftmaxKlass(trainMat, wMat, bVec); 
% Angleichen der Kategorien
trainLblPred = categorical(trainLblPred, categories(trainLblPred), categories(trainLbl));
errTrain = sum(trainLblPred ~= trainLbl) / length(trainLbl);

[testLblPred, pMatTest] = applySoftmaxKlass(testMat, wMat, bVec); 
testLblPred = categorical(testLblPred, categories(testLblPred), categories(testLbl));
errTest = sum(testLblPred ~= testLbl) / length(testLbl);





%% 3.4.2) Verlauf des stochastischen Gradientenabstiegs
sgaIts = size(track, 2);   %% Anzahl sga-Iterationen 
xsVec = zeros(sgaIts, 1); 
normGradVec = zeros(sgaIts, 1);
tic
for i = 1:sgaIts
   xsVec(i) = xs(track(:,i)); 
   normGradVec(i)  = norm(dxs(track(:,i))); 
end
toc

figure('WindowStyle', 'docked'); 
yyaxis left; 
plot(1:sgaIts, xsVec);
ylabel('XS(w,b)'); 
yyaxis right; 
plot(1:sgaIts, normGradVec);
ylabel('| \nabla XS(w,b)|'); 
xlabel('Iterationen'); 
title('Verlauf der Kostenfunktion (stoc. GA)')


%% 3.5) Jetzt mit Matlab's Deep learning Toolbox
layers = [featureInputLayer(P)
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];

options = trainingOptions('sgdm', ...
   'MiniBatchSize', 128, ...
   'ValidationData', {testMat, testLbl}, ...
   'MaxEpochs',20, ...
   'Verbose', false, ...
   'Plots', 'training-progress'); 
   % 'Plots', 'Training-Progress');

%% 3.5.1) Der Trainingsprozess
tic
[net, info] = trainNetwork(trainMat, trainLbl, layers, options);
toc
%% 3.5.2) Trainings- und Testfehler
trainLblPred = classify(net, trainMat); 
errTrainML = sum(testLbl ~= testLblPred)/length(testLbl); 

testLblPred = classify(net, testMat); 
errTestML = sum(testLbl ~= testLblPred)/length(testLbl); 

%% Bsp 4: Xor-Daten
% Daten erzeugen und plotten
load(fullfile('..', 'Datensaetze', 'Xor2D.mat'));

xdim = 1.1*[min(X(:,1)), max(X(:,1))]; 
ydim = 1.1*[min(X(:,2)), max(X(:,2))];
fD = figure('WindowStyle', 'docked'); 
gscatter(X(:,1), X(:,2), lbl, 'rg', '..', 15*[1,1]);
title('Trainingsdaten'); 
legend off; 
axis([xdim, ydim]); 

[N, P] = size(X); 
C = length(categories(lbl));   % Anzahl Klassen

%% 4.1) Softmax-Klassifizierung
lambda = 1;
[~, dxs] = xsSoftmaxKlass(X, lbl, lambda);
w0 = 0.001 * randn(P,C);
b0 = 0.0 * rand(1, C);
wbVec0 = [w0(:); b0'];
nIts = 500;
eta = 0.005; 

% Trainieren und Anwenden
wbopt = gaRmsProp(dxs, eta, wbVec0, nIts, 0.95); 
wMat = reshape(wbopt(1:(P*C)), P, C);
bVec = reshape(wbopt((C*P+1):end), 1, C);
lblPred = applySoftmaxKlass(X, wMat, bVec); 
% lblPred = categorical(lblPred, 1:C, categories(lbl));
err = sum(lblPred ~= lbl) / length(lbl); 

%% 4.1.1) Klassifizierungsregionen
nGrid = 100; 
[Xg,Yg] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
   linspace(ydim(1), ydim(2), nGrid));
Dgrid = [Xg(:), Yg(:)];
[~, pGrid]  = applySoftmaxKlass(Dgrid, wMat, bVec); 
img = zeros([size(Xg), 3]);
img(:,:,1) = reshape(pGrid(:,1), size(Xg));
img(:,:,2) = 1 - img(:,:,1);
% Eintragen in den Scatterplot:
hold on;
image(img, 'XData', xdim, 'YData', ydim);
% Trennlinien
thres = 0.5;
contour(Xg,Yg,img(:,:,1), thres*[1,1], 'k', 'Linewidth', 2, 'DisplayName', 'Trennlinie'); 
alpha(0.5);  % Transparenz des Hintergrundes 
hold off; 
axis([xdim, ydim]);


%% 4.2) Klassifizierung mit DL-Toolbox

layers = [featureInputLayer(P)
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];

options = trainingOptions('rmsprop',...
   'ValidationData', {X, lbl}, ...
   'MaxEpochs', 1000, ...
   'MinibatchSize', N, ...
   'Verbose', false, ...
   'L2Regularization', 0.01, ...
   'Plots', 'training-progress'); 
net = trainNetwork(X, lbl, layers, options);

%% 4.3) Quadratisches Modell 
X2 = [X, X.^2, X(:,1).*X(:,2)];
P = size(X2, 2);
lambda = 1;
[xs, dxs] = xsSoftmaxKlass(X2, lbl, lambda);
w0 = 0.001 * randn(P,C);
b0 = 0.0 * rand(1, C);
wbVec0 = [w0(:); b0'];
nIts = 500;
eta = 0.005; 
 
tic
[wbopt, track] = gaRmsProp(dxs, eta, wbVec0, nIts, 0.95); 
toc

wMat = reshape(wbopt(1:(P*C)), P, C);
bVec = reshape(wbopt((C*P+1):end), 1, C);
lblPred = applySoftmaxKlass(X2, wMat, bVec); 
err = sum(lblPred ~= lbl) / length(lbl); 

%% 4.3.2) Klassifizierungsregionen
figure('WindowStyle', 'docked'); 
gscatter(X(:,1), X(:,2), lbl, 'rg', '..', 15*[1,1]);
title('Trainingsdaten'); 
legend off; 
axis([xdim, ydim]); 
D2grid = [Dgrid, Dgrid.^2, Dgrid(:,1).*Dgrid(:,2)];
[~, pGrid]  = applySoftmaxKlass(D2grid, wMat, bVec); 
img = zeros([size(Xg), 3]);
img(:,:,1) = reshape(pGrid(:,1), size(Xg));
img(:,:,2) = 1 - img(:,:,1);
% Eintragen in den Scatterplot:
hold on;
image(img, 'XData', xdim, 'YData', ydim);
% Trennlinien
thres = 0.5;
contour(Xg,Yg,img(:,:,1), thres*[1,1], 'k', 'Linewidth', 2, 'DisplayName', 'Trennlinie'); 
alpha(0.5);  % Transparenz des Hintergrundes 
hold off; 
axis([xdim, ydim]);