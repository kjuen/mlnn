%% Einwohnerdaten Hamburg: 
% Datenquelle: Wikipedia, https://de.wikipedia.org/wiki/Einwohnerentwicklung_von_Hamburg

%% Daten einlesen und plotten
T = readtable(fullfile('..', 'Datensaetze', 'EinwohnerHamburg.csv'));
% Fehlende Werte rausschmeissen
m = ~isnan(T.Melderegister);
Jahr = T.Jahr(m);
Einwohner = T.Melderegister(m);
f1 = figure; 
scatter(Jahr, Einwohner/1000, 'Displayname', 'Daten Melderegister');
xlabel('Jahr');
ylabel('Anzahl Einwohner in 1000');
grid on; 
title('Bevölkerungsentwicklung Hamburg');
