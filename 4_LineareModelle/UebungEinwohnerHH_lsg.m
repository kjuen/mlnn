%% Einwohnerdaten Hamburg:
% Datenquelle: Wikipedia, https://de.wikipedia.org/wiki/Einwohnerentwicklung_von_Hamburg

%% Daten einlesen und plotten
T = readtable(fullfile('..', 'Datensaetze', 'EinwohnerHamburg.csv'));
m = ~isnan(T.Melderegister);
Jahr = T.Jahr(m);
Einwohner = T.Melderegister(m);
f1 = figure;
scatter(Jahr, Einwohner/1000, 'Displayname', 'Daten Melderegister');
xlabel('Jahr');
ylabel('Anzahl Einwohner in 1000');
grid on;
title('Bevölkerungsentwicklung Hamburg');

%% Polynom erster und zweiter Ordnung ohne Regularisierung
D1 = [Jahr, ones(size(Jahr))];
wb1 = linsolve(D1'*D1, D1'*Einwohner);
fit1 = @(j) wb1(1) * j + wb1(2);
D2 = [Jahr.^2, Jahr, ones(size(Jahr))];
wb2 = linsolve(D2'*D2, D2'*Einwohner);
fit2 = @(j) wb2(1) * j.^2 + wb2(2) * j + wb2(3);
hold on;
plot([Jahr; 2025], fit1([Jahr; 2025])/1000, ...
   'DisplayName', 'Ordn 1');
plot([Jahr; 2025], fit2([Jahr; 2025])/1000, ...
   'DisplayName', 'Ordn 2');
hold off;
legend('Location','NW');

%% Daten normieren
% mJ = 0; sJ = 1; mE = 0; sE = 1;
mJ = mean(Jahr);
sJ = std(Jahr);
Jn = (Jahr - mJ) / sJ;
mE = mean(Einwohner);
sE = std(Einwohner);
En = (Einwohner - mE) / sE;
f2 = figure;
scatter(Jn, En);
xlabel('Jahr (normiert)');
ylabel('Anzahl Einwohner (normiert)');
grid on;
title('Bevölkerungsentwicklung Hamburg (normiert)');

%% Polynom mit den normierten Daten fitten
D1n = [Jn, ones(size(Jn))];
wb1n = linsolve(D1n'*D1n, D1n'*En);
fit1n = @(j) wb1n(1) * j + wb1n(2);
D2n = [Jn.^2, Jn, ones(size(Jn))];
wb2n = linsolve(D2n'*D2n, D2n'*En);
fit2n = @(j) wb2n(1) * j.^2 + wb2n(2) * j + wb2n(3);
hold on;
plot(Jn, fit1n(Jn), 'DisplayName', 'Ordn 1');
plot(Jn, fit2n(Jn), 'DisplayName', 'Ordn 2');
hold off;
legend('Location','NW');
