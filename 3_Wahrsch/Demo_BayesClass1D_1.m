%% Bayes Classfikator 1D

load(fullfile('..', 'Datensaetze', 'Data1D2Gr_1.mat')); 

t = tiledlayout(2,1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile(1);
gscatter(xTrain, 0*trainLbl, trainLbl, 'rb','ox'); 
hold on; 
scatter(xTest, 0.5*ones(size(xTest)), 'k'); 
hold off; 
xlim([-4,4]); 
ylim([-0.5, 1.5]); 
set(gca, 'YTick', []); 
xlabel('x'); 
legend({'Train: lbl=1', 'Train: lbl=2', 'Test'}); 

%% Schaetze Priors aus den Trainingsdaten
Pri1 = sum(trainLbl==1)/length(trainLbl);
Pri2 = sum(trainLbl==2)/length(trainLbl); 


%% Schaetze class-conditionals als Normalverteilung
m1 = trainLbl == 1;
m2 = trainLbl == 2; 
mu1 = mean(xTrain(m1)); 
s1 = std(xTrain(m1));
mu2 = mean(xTrain(m2)); 
s2 = std(xTrain(m2)); 

cc1 = @(x) 1/(sqrt(2*pi)*s1) * exp(-1/2*(x-mu1).^2/(s1^2));
cc2 = @(x) 1/(sqrt(2*pi)*s2) * exp(-1/2*(x-mu2).^2/(s2^2));  
xx = linspace(-4,4); 

nexttile(2);
plot(xx, cc1(xx), '--r', xx, cc2(xx), '--b'); 
hold on; 
plot(xx, Pri1*cc1(xx), 'r', xx, Pri2*cc2(xx), 'b'); 
hold off; 
xlim([-4,4]); 
legend('Class Conditional', 'Class Conditional', 'Posterior', 'Posterior', ...
   'Location', 'NorthWest') ; 

%% Berechnung Trainingsfehler
trainPost = [cc1(xTrain)*Pri1, cc2(xTrain)*Pri2];
trainLblPred = zeros(size(trainPost, 1), 1);
for nn = 1:length(trainLblPred)
   [~, idx] = max(trainPost(nn,:)); 
   trainLblPred(nn) = idx;
end
trainErr = sum(trainLblPred ~= trainLbl) / length(trainLblPred);
% Uebrigens, so gehts kuerzer: 
[~, trainLblPred] = max(trainPost, [], 2); 

%% Anwenden auf Testdaten
% Berechne unnormierte Posteriors: 

% Pri1 = 0.5; 
% Pri2 = 0.5; 
testPost = [cc1(xTest)*Pri1, cc2(xTest)*Pri2];
testLblPred = zeros(size(testPost, 1), 1);
for nn = 1:length(trainLblPred)
   [~, idx] = max(testPost(nn,:)); 
   testLblPred(nn) = idx;
end
testErr = sum(testLblPred ~= testLbl) / length(testLblPred);
nexttile(1); 
hold on; 
gscatter(xTest, ones(size(testLblPred)), testLblPred, 'rb','ox'); 
hold off; 

%% Das gleiche mit Matlabs fitcnb
Mdl = fitcnb(xTrain, trainLbl); 
% per default werden Normalverteilungen angenommen
[trainLblPred2, trainPost2] = predict(Mdl, xTrain); 
trainErr2 = sum(trainLblPred2 ~= trainLbl) / length(trainLblPred);
testLblPred2 = predict(Mdl, xTest); 
testErr2 = sum(testLblPred2 ~= testLbl) / length(testLblPred);
% disp(Mdl.DistributionNames); 
% Vgl. z.B. Verteilungsparameter von Gr1
% disp([Mdl.DistributionParameters{1,1}, [mu1; s1]]); 




%% Nochmal die Schaetzung der bedingten Wahrscheinlichkeiten
% (class conditionals)
close all; 
yyaxis left;
mask = trainLbl==1; 
h1 = histogram(xTrain(mask), 20, 'EdgeColor', 'red', 'FaceColor', 'red', 'FaceAlpha', 0.3);
hold on; 
h2 = histogram(xTrain(~mask), 20, 'EdgeColor', 'blue', 'FaceColor', 'blue', 'FaceAlpha', 0.3);
hold off; 
yyaxis right; 
p1 = plot(xx, cc1(xx), '--r', xx, cc2(xx), '--b'); 
xlim([-4,4]); 
title('Schaetzung der Wahrscheinlichkeitsdichten');
legend('1', '2', 'Gauss', 'Gauss'); 


%% Schaetzung mit Fenstermethode
cc1Win = @(x, bw) ksdensity(xTrain(trainLbl==1), x, 'Bandwidth', bw); 
cc2Win = @(x, bw) ksdensity(xTrain(trainLbl==2), x, 'Bandwidth', bw); 
hold on; 
plot(xx, cc1Win(xx, 0.5), '-r', 'DisplayName', 'Kernel'); 
plot(xx, cc2Win(xx, 0.25), '-b', 'DisplayName', 'Kernel');
hold off; 


