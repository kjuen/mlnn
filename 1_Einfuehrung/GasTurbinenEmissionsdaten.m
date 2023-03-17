%% Gas Turbine CO and NOx Emission Data Set Data Set
% http://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set

%%
% Merkmale:
%
% * Ambient temperature (AT): C 
% * Ambient pressure (AP) mbar
% * Ambient humidity (AH) %
% * Air filter difference pressure (AFDP) mbar
% * Gas turbine exhaust pressure (GTEP) mbar
% * Turbine inlet temperature (TIT) C
% * Turbine after temperature (TAT) C
% * Compressor discharge pressure (CDP) mbar
% * Carbon monoxide (CO) mg/m3
% * Nitrogen oxides (NOx) mg/m3
%
% Zielgroesse:
%
% * Turbine energy yield (TEY) MWH
%

%% Daten aus dem Jahr 2013 zur Modellentwicklung
T = readtable(fullfile('..', 'Datensaetze', 'gt_2013.csv')); 
% Zielgroeße
tey13 = table2array(T(:,8));
T(:,8) = [];
P = size(T, 2);   % Anzahl Merkmale
% Daten und Designmatrix:
M = table2array(T); 
D13 = [ones(size(tey13)), M];
% Normalengleichungen:
bw = linsolve(D13'*D13, D13'*tey13);

tey13Pred = D13 * bw; 
RMSE13 = sqrt(1/length(tey13) * sum( (tey13 - tey13Pred).^2 )); 
scatter(tey13, tey13Pred, '.'); 
title('Training: Turbine energy yield (TEY), 2013', sprintf('RMSE = %.2f', RMSE13)); 
xlabel('Zielgroesse TEY: Gemessen'), ylabel('Zielgroesse TEY: Vorhersage'); 

%% Daten aus dem Jahr 2014 zur Überprüfung des Modells
% Das Modell wird auf diese Daten angewandt und die vorhergesagten Werte
% mit den gemessenen verglichen
T = readtable(fullfile('..', 'Datensaetze', 'gt_2014.csv')); 
tey14 = table2array(T(:,8));
T(:,8) = [];
M = table2array(T); 
D14 = [ones(size(tey14)), M];
tey14Pred = D14*bw; 
disp([tey14Pred(1:10), tey14(1:10)]); 

RMSE14 = sqrt(1/length(tey14) * sum( (tey14 - tey14Pred).^2 )); 
scatter(tey14, tey14Pred, '.'); 

title('Training: Turbine energy yield (TEY), 2014', sprintf('RMSE = %.2f', RMSE14)); 
xlabel('Zielgroesse TEY: Gemessen'), ylabel('Zielgroesse TEY: Vorhersage'); 

%% Histogramm der Abweichungen
histogram(tey13Pred - tey13, 30); 
hold on; 
histogram(tey14Pred - tey14, 30); 
hold off; 
legend('2013 (Training)', '2014 (Test)');
xlabel('Abweichungen'), ylabel('Anzahl'); 
title('Histogramm der Abweichungen'); 
