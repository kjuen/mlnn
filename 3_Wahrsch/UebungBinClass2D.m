% Uebungsaufgabe Kapitel 3

%% Daten laden und plotten
dateiName = fullfile('..', 'Datensaetze', 'binclass2D.csv');
T = readtable(dateiName); 
disp(head(T)); 
% Speichern als Datenmatrix und kategorielles Array
trainMat = [T.x, T.y]; 
trainLbl = categorical(T.Klasse);
tabulate(trainLbl); 
gscatter(trainMat(:,1), trainMat(:,2), trainLbl, 'rg','o'); 
title('Trainingsdaten'); 
