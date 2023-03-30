%% Demo 1-D-Regression und Regularisierung

%% Daten laden, normieren und plotten
filename = fullfile('..', 'Datensaetze', 'DemoRegularisierungData.mat'); 
load(filename); 
% Normieren der Daten
m  = mean(xTrain); 
s = std(xTrain); 
xTrain = (xTrain - m)/s;
xTest = (xTest - m)/s; 
nTrain = size(xTrain, 1);
% Plotten
plot(xTrain,yTrain, 'o', 'DisplayName', 'Trainingsdaten');
hold on;
plot(xTest,yTest, 'x', 'DisplayName', 'Testdaten'); 
hold off;
xlabel('x'), ylabel('y'); 
xlim(minmax(xTest')); 
ylim(5*[-1, 1]); 
legend;

%% Polynom anpassen
xx = linspace(1.1*min(xTest), 1.1*max(xTest));   % x-Diskretisierung zum Plotten
D = [];    % Design-Matrix
F = 2; 
for k=0:F
   D = [xTrain.^k, D];  %#ok<AGROW>
end
wb = linsolve(D'*D, D'*yTrain); 
yFit = polyval(wb, xx); 
hold on; 
plot(xx, yFit, 'Displayname', sprintf('Fit-Poly, Ordn %i', F)); 
hold off; 
legend;

yFitTrain = polyval(wb, xTrain);
RMSETrain = sqrt(mean( (yTrain-yFitTrain).^2 ));
yFitTest = polyval(wb, xTest);
RMSETest = sqrt(mean( (yTest-yFitTest).^2 )); 
st = sprintf('RMSE-Train=%.3f, RMSE-Test=%.3f', RMSETrain, RMSETest); 
title(sprintf('Polynom der Ordung %i', F), st); 

%% Das gleiche mit Regularisierung
lambda = 1; 
wb = linsolve(D'*D+lambda*eye(F+1), D'*yTrain); 
yFit = polyval(wb, xx); 
hold on; 
plot(xx, yFit, 'Displayname', sprintf('Fit-Poly mit Regul., Ordn. %i', F)); 
hold off; 
legend;
% 
yFitTrain = polyval(wb, xTrain);
RMSETrainReg = sqrt(mean( (yTrain-yFitTrain).^2 ));
yFitTest = polyval(wb, xTest);
RMSETestReg = sqrt(mean( (yTest-yFitTest).^2 ));
streg = {st; sprintf('Mit Regul.: RMSE-Train=%.3f, RMSE-Test=%.3f', ...
   RMSETrainReg, RMSETestReg)}; 
title(sprintf('Polynom der Ordung %i', F), streg); 

%% Loop über die Polynomordnung
lambda = 0.2;
Fvec = 1:12; 
RMSETrainVec = zeros(size(Fvec));  
RMSETestVec = zeros(size(Fvec));
RMSETrainVecReg = zeros(size(Fvec));  
RMSETestVecReg = zeros(size(Fvec)); 
for ff = 1:length(Fvec)
   F = Fvec(ff);
   % Polynom fitten wie oben
   D = [];     % Design-Matrix
   for k=0:F
      D = [xTrain.^k, D];  %#ok<AGROW>
   end
   % Berechnung von RMSE ohne Regularisierung
   wb = linsolve(D'*D, D'*yTrain);
      
   yFitTrain = polyval(wb, xTrain);
   RMSETrainVec(ff) = sqrt(mean( (yTrain-yFitTrain).^2 ));
   yFitTest = polyval(wb, xTest);
   RMSETestVec(ff) = sqrt(mean( (yTest-yFitTest).^2 ));
   
   % Berechnung von RMSE mit Regularisierung
   wbReg = linsolve(D'*D + lambda*eye(F+1), D'*yTrain);
   yFitTrain = polyval(wbReg, xTrain);
   RMSETrainVecReg(ff) = sqrt(mean( (yTrain-yFitTrain).^2 ));
   yFitTest = polyval(wbReg, xTest);
   RMSETestVecReg(ff) = sqrt(mean( (yTest-yFitTest).^2 ));
end
%
plot(Fvec, RMSETrainVec, 'o-'); 
hold on; 
plot(Fvec, RMSETestVec, 'o-');
plot(Fvec, RMSETrainVecReg, 'o-');
plot(Fvec, RMSETestVecReg, 'o-');
hold off; 
legend('Train', 'Test', 'Train (Reg)', 'Test (Reg)', ...
   'Location', 'NorthWest'); 
axis([Fvec(1), Fvec(end), 0, 2]);
xlabel('Polynomordnung'); 
ylabel('MSE'); 
title('MSE in Abh. der Polynormordnung', ...
   sprintf('Regularisierung: \\lambda = %.2f', lambda)); 