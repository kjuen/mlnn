%% Beispiel fuer Normierung der Daten

load(fullfile('..', 'Datensaetze', 'DemoNormierungDaten.mat'));
gscatter(D(:,1), D(:,2), lbl); 
%%
kNN = fitcknn(D, lbl, 'NumNeighbors',11);
lblNeu = predict(kNN, D); 
err = sum(lblNeu ~= lbl) / length(lbl); 
disp(err);
title(sprintf('Originaldaten: Fehlerrate = %.1f%%', 100*err)); 

%% Skaliere die Merkmale!
m = mean(D);
sigma = std(D);
Ds = (D-m)./sigma; 
figure;
gscatter(Ds(:,1), Ds(:,2), lbl); 

kNN = fitcknn(Ds, lbl, 'NumNeighbors',11);
lblNeu = predict(kNN, Ds); 
err2 = sum(lblNeu ~= lbl) / length(lbl); 
disp(err2); 
title(sprintf('Normierte Daten: Fehlerrate = %.1f%%', 100*err2)); 
