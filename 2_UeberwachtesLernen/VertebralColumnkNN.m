%% Vertebral Column (=Wirbelsäule) Datensatz: kNN-Analyse
% https://archive.ics.uci.edu/ml/datasets/Vertebral+Column
% Zielgroessen: Orthopaedische Klassen: DH (Disk Hernia), Spondylolisthesis (SL), Normal (NO) and Abnormal (AB)
% Der Datensatz 2C enthaelt die Klassen AB und NO und der andere die
% Klassen DH, SL und NO. 
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
nRuns = 20;
kVec  = [1:2:51, 56:5:101]; 
nk = length(kVec);
meanTestErr = zeros(nk,1); 
meanTrainErr = zeros(nk,1); 
tic;
for kk=1:nk
    disp(kVec(kk))
    trainFunc = @(d,l) fitcknn(d,l, 'NumNeighbors', kVec(kk));
    [xTestErr, xTrainErr] = xval(trainFunc, @predict, ...
       D, lbl, nChunks, nRuns);
    meanTestErr(kk) = mean(xTestErr); 
    meanTrainErr(kk) = mean(xTrainErr); 
end
toc

%% Graphisch darstellen
plot(kVec, meanTestErr, 'r', kVec, meanTrainErr, 'b', 'Linewidth', 2); 
axis tight; 
legend('Testfehler', 'Trainingsfehler', 'Location', 'SouthEast'); 
ylim([0, 0.3]);
xlabel('Anzahl Nachbarn'); 
ylabel('Fehlerrate'); 
[errOpt, idx] = min(meanTestErr); 
kOpt = kVec(idx);
title('Kreuzvalidierung Naive Bayes', ...
   sprintf('Bester Testfehler: %.3f bei k=%i', errOpt, kOpt)); 

%% Nochmal mit ganzem Datensatz 
knn = fitcknn(D, lbl, 'NumNeighbors', kOpt);
lblPred = predict(knn, D); 
trainErr = sum(lblPred ~= lbl) / length(lbl); 
