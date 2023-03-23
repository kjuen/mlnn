%% Vertebral Column (=Wirbelsäule) Datensatz: Naive-Bayes-Analyse
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

nb = fitcnb(D, lbl, 'DistributionNames', 'kernel', 'Width', 0.5); 
lblPred = predict(nb, D); 
trainErr = sum(lblPred ~= lbl) / length(lbl); 

%% Besten Wert fuer Width-Parameter finden
addpath(fullfile('..', '2_UeberwachtesLernen')); % fuer xval
nChunks = 5;   % mit 80% wird jeweils trainiert
nRuns = 20;
bwVec  = [1:10, 12:2:30, 35:5:100]/100;
nbw = length(bwVec);
meanTestErr = zeros(nbw,1); 
meanTrainErr = zeros(nbw,1); 
tic;
for kk=1:nbw
    disp(bwVec(kk))
    trainFunc = @(d,l) fitcnb(d,l, 'DistributionNames', 'kernel', 'Width', bwVec(kk)); % bwVec(kk)); 
    [xTestErr, xTrainErr] = xval(trainFunc, @predict, ...
       D, lbl, nChunks, nRuns);
    meanTestErr(kk) = mean(xTestErr); 
    meanTrainErr(kk) = mean(xTrainErr); 
end
toc

%% Graphisch darstellen
plot(bwVec, meanTestErr, 'r', bwVec, meanTrainErr, 'b', 'Linewidth', 2); 
axis tight; 
legend('Testfehler', 'Trainingsfehler', 'Location', 'SouthEast'); 
ylim([0, 0.3]);
xlabel('Fensterbreite'); 
ylabel('Fehlerrate'); 
[errOpt, idx] = min(meanTestErr); 
bwOpt = bwVec(idx);
title('Kreuzvalidierung Naive Bayes', ...
   sprintf('Bester Testfehler: %.2f bei bw=%.2f', errOpt, bwOpt)); 

%% Nochmal mit ganzem Datensatz 
nb = fitcnb(D, lbl, 'DistributionNames', 'kernel', 'Width', bwOpt); 
lblPred = predict(nb, D); 
trainErr = sum(lblPred ~= lbl) / length(lbl); 