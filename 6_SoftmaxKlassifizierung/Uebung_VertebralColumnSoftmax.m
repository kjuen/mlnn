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

%% Nomierung und Histogram der 6 Merkmale
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

%% Anwenden des Gradientenabstiegs
addpath(fullfile('..', '5_Gradientenabstieg'));  % fuer ga-Funktionen
P = 6; 
C = 3; 
% hier weiter machen ...