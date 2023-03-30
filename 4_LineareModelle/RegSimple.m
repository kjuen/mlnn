%% Ein einfaches Beispiel für Regression 

% func = @(x) - 0.2*x.^2 + 1.1*x - 0.75;
func = @(x)  + 1.1*x - 0.75;
addNoise = @(x) x + 0.2 * randn(size(x)); 

%% Trainingsdaten erzeugen
nTrain = 100; 
xTrain = sort(5*rand(nTrain, 1)); 
yTrain = addNoise(func(xTrain));
scatter(xTrain, yTrain, 'DisplayName', 'Trainingsdaten'); 
xlabel('x'), ylabel('y'); 
t = 'Einfaches Regressionsbeispiel';
title(t);

%% Methode der kleinsten Quadrate
DTrain = [xTrain, ones(nTrain, 1)]; 
wb = linsolve(DTrain'*DTrain, DTrain'*yTrain); 
zTrain = DTrain*wb; 
hold on 
plot(xTrain, zTrain, ...
   'DisplayName', sprintf('Fit-Poly, Ordn, %i', size(DTrain,2)-1)); 
hold off;
legend('Location', 'NW'); 
MSETrain = mean( (zTrain-yTrain).^2 );
RMSETrain = sqrt(MSETrain);
title(t, sprintf('RMSE=%.2f', RMSETrain)); 

