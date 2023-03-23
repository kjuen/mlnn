%% Noch ein Datensatz
load(fullfile('..', 'Datensaetze', 'Data1D2Gr_2.mat')); 
xdim = [-2, 6]; 

%% Plot der Trainingsdaten
t = tiledlayout(4,1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile(1);
% subplot(4,1,1); 
gscatter(xTrain, 0*trainLbl, trainLbl, 'rb','o'); 
title('Trainingsdaten'); 
axis([xdim, -0.5, 0.5]); 
set(gca, 'YTick', []), xlabel(''); 
legend off; 
%% Plot der Testdaten
nexttile(2);
% subplot(4,1,2); 
gscatter(xTest, 0*testLbl, testLbl, 'rb', 'o'); 
hold off; 
axis([xdim, -0.5, 0.5]); 
set(gca, 'YTick', []), xlabel(''); 
title('Testdaten'); 
legend off; 
%% Schaetze Priors aus den Trainingsdaten
Pri1 = sum(trainLbl==1)/length(trainLbl);
Pri2 = sum(trainLbl==2)/length(trainLbl); 


%% Schaetze class-conditionals als Normalverteilung
% Fazit: Schwache Performance, da die class conditionals nicht gut
% angepasst werden.
m1 = trainLbl == 1;
m2 = trainLbl == 2; 
mu1 = mean(xTrain(m1)); 
s1 = std(xTrain(m1));
mu2 = mean(xTrain(m2)); 
s2 = std(xTrain(m2)); 

cc1 = @(x) 1/(sqrt(2*pi)*s1) * exp(-1/2*(x-mu1).^2/(s1^2));
cc2 = @(x) 1/(sqrt(2*pi)*s2) * exp(-1/2*(x-mu2).^2/(s2^2));  

nexttile([2,1]);
yyaxis left;
mask = trainLbl==1; 
histogram(xTrain(mask), 20, 'EdgeColor', 'red', 'FaceColor', 'red', 'FaceAlpha', 0.3);
hold on; 
histogram(xTrain(~mask), 20, 'EdgeColor', 'blue', 'FaceColor', 'blue', 'FaceAlpha', 0.3);
hold off; 
xlabel('x');
yyaxis right; 

xx = linspace(xdim(1), xdim(2));
plot(xx, cc1(xx), 'r-', xx, cc2(xx), 'b-'); 
xlim(xdim); 
legend('Gr 1', 'Gr 2', 'cc1', 'cc2', ...
   'Location', 'NorthWest') ; 

% Berechnung Trainings- und Testfehler
postTrain = [cc1(xTrain)*Pri1, cc2(xTrain)*Pri2];
[~, trainLblPred] = max(postTrain, [], 2); 
trainErr = sum(trainLblPred ~= trainLbl) / length(trainLblPred);
postTest = [cc1(xTest)*Pri1, cc2(xTest)*Pri2];
[~, testLblPred] = max(postTest, [], 2); 
testErr = sum(testLblPred ~= testLbl) / length(testLblPred);
title('Class conditionals als Normalverteilungen', ...
   sprintf('Train-Fehler=%.1f%%, Test-Fehler=%.1f%%', trainErr*100, testErr*100));

%% Alternativ: Schaetzung mit der Fenstermethode
% Fazit: Die Fehlerrate sinkt deutlich, denn die class-conditionals werden
% jetzt besser modelliert
cc1 = @(x) ksdensity(xTrain(trainLbl==1), x, 'Bandwidth', 0.4);
cc2 = @(x, bw) ksdensity(xTrain(trainLbl==2), x, 'Bandwidth', 0.4);
yyaxis left;
mask = trainLbl==1; 
histogram(xTrain(mask), 20, 'EdgeColor', 'red', 'FaceColor', 'red', 'FaceAlpha', 0.3);
hold on; 
histogram(xTrain(~mask), 20, 'EdgeColor', 'blue', 'FaceColor', 'blue', 'FaceAlpha', 0.3);
hold off; 
xlabel('x');
yyaxis right; 

xx = linspace(xdim(1), xdim(2));
plot(xx, cc1(xx), 'r-', xx, cc2(xx), 'b-'); 
xlim(xdim); 
legend('Gr 1', 'Gr 2', 'cc1', 'cc2', ...
   'Location', 'NorthWest') ; 

% Berechnung Trainings- und Testfehler
postTrain = [cc1(xTrain)*Pri1, cc2(xTrain)*Pri2];
[~, trainLblPred] = max(postTrain, [], 2); 
trainErr = sum(trainLblPred ~= trainLbl) / length(trainLblPred);
postTest = [cc1(xTest)*Pri1, cc2(xTest)*Pri2];
[~, testLblPred] = max(postTest, [], 2); 
testErr = sum(testLblPred ~= testLbl) / length(testLblPred);
title('Class conditionals mit der Fenstermethode', ...
   sprintf('Train-Fehler=%.1f%%, Test-Fehler=%.1f%%', trainErr*100, testErr*100));


%% Hyperparametersuche: Fensterbreite
% Suche nach der passenden Fensterbreite
cc1Win = @(x, bw) ksdensity(xTrain(trainLbl==1), x, 'Bandwidth', bw); 
cc2Win = @(x, bw) ksdensity(xTrain(trainLbl==2), x, 'Bandwidth', bw); 
bwVec = 0.01:0.01:0.8; 
trainErrVec = zeros(size(bwVec)); 
testErrVec = zeros(size(bwVec));
for kk = 1:length(bwVec)
   bw = bwVec(kk); 
   % Trainingsfehler
   post = [cc1Win(xTrain, bw)*Pri1, cc2Win(xTrain, bw)*Pri2];
   [~, lblPred] = max(post, [], 2); 
   trainErrVec(kk) = sum(lblPred ~= trainLbl) / length(lblPred);
   % Testfehler
   post = [cc1Win(xTest, bw)*Pri1, cc2Win(xTest, bw)*Pri2];
   [~, lblPred] = max(post, [], 2); 
   testErrVec(kk) = sum(lblPred ~= testLbl) / length(lblPred);
end
%%
figure; 
plot(bwVec, testErrVec, 'r', bwVec, trainErrVec, 'b', 'Linewidth', 2); 
legend('Testfehler', 'Trainingsfehler', 'Location', 'Southeast'); 
xlabel('Bandbreite'); 
ylabel('Fehlerrate'); 
[minErr, idx] = min(testErrVec);
title('Fehlerrate in Abhaengigeit von der Fensterbreite', ...
   sprintf('Bester Testfehler %.1f%% bei Fensterbreite %.2f', 100*minErr, bwVec(idx)));
