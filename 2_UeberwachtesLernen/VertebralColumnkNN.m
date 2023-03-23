%% Vertebral Column (=Wirbelsäule) Datensatz: kNN-Analyse
% https://archive.ics.uci.edu/ml/datasets/Vertebral+Column
% Zielgroessen: Orthopaedische Klassen: DH (Disk Hernia), SL (Spondylolisthesis), NO (Normal)
% 6 biomechanische Messwerte als Merkmale (das letzte ist die Klasse)

fileName = fullfile('..', 'Datensaetze', 'VertebralColumn.csv'); 
T = readtable(fileName); 
disp(head(T));

D = table2array(T(:,1:6));
lbl = categorical(T.orthopaedicClass);
tabulate(lbl);

%% Die Merkmale
t = tiledlayout(3,2, 'TileSpacing','compact', 'Padding', 'compact'); 
for ii = 1:6
   nexttile
   histogram(D(:,ii));
   title(T.Properties.VariableNames{ii}); 
end
title(t, 'Rohdaten');

%% Normierung der Daten
m = mean(D); 
sigma = std(D); 
D = (D-m)./sigma;

t = tiledlayout(3,2, 'TileSpacing','compact', 'Padding', 'compact'); 
for ii = 1:6
   nexttile
   histogram(D(:,ii));
   title(T.Properties.VariableNames{ii});
end
title(t, 'Normierte Daten');

%% Trainingsfehler

knn = fitcknn(D, lbl, 'NumNeighbors', 7);
lblPred = predict(knn, D); 
trainErr = sum(lblPred ~= lbl) / length(lbl); 

%% Besten Wert fuer k finden

nChunks = 5;   % mit 80% wird jeweils trainiert
nRuns = 10;
kVec  = [1:2:51, 56:5:101]; 
nk = length(kVec);
meanXValErr = zeros(nk,1); 
meanTrainErr = zeros(nk,1); 
tic;
for kk=1:nk
    disp(kVec(kk))
    trainFunc = @(d,l) fitcknn(d,l, 'NumNeighbors', kVec(kk));
    [xTestErr, xTrainErr] = xval(trainFunc, @predict, ...
       D, lbl, nChunks, nRuns);
    meanXValErr(kk) = mean(xTestErr); 
    meanTrainErr(kk) = mean(xTrainErr); 
end
toc

%% Graphisch darstellen
plot(kVec, meanXValErr, 'r', kVec, meanTrainErr, 'b', 'Linewidth', 2); 
axis tight; 
legend('Xval-Fehler', 'Trainingsfehler', 'Location', 'SouthEast'); 
ylim([0, 0.3]);
xlabel('Anzahl Nachbarn'); 
ylabel('Fehlerrate'); 
[errOpt, idx] = min(meanXValErr); 
kOpt = kVec(idx);
title('Kreuzvalidierung kNN', ...
   sprintf('Bester Testfehler: %.2f%% bei k=%i', 100*errOpt, kOpt)); 

%% Nochmal mit ganzem Datensatz 
knn = fitcknn(D, lbl, 'NumNeighbors', kOpt);
lblPred = predict(knn, D); 
trainErr = sum(lblPred ~= lbl) / length(lbl); 
