%% Winkelerkennung von gedrehten Ziffern

%% Daten laden und ein paar Bilder darstellen
filename = fullfile('..', 'Datensaetze', 'GedrehteZweien.mat');
load(filename); 
nTrain = numel(YTrain);
nTest = numel(YTest); 
imSize = size(XTrain, 1:2); 

%
idx = randperm(nTrain,20);
t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
title(t, 'Ein paar Trainingsbilder');
for i = 1:numel(idx)
    nexttile;
    imshow(XTrain(:,:,:,idx(i)));
    title(sprintf('%i°', YTrain(idx(i))));
end
%% Histogramm der Winkel der Trainingsdaten
figure
histogram(YTrain, 20)
axis tight
ylabel('Anzahl'), xlabel('Winkel (Grad)');
title('Histogramm der Winkel der Trainingsdaten');

%% Lineares Modell anpassen
P = imSize(1)*imSize(2);   % Anzahl Merkmale
% Designmatrix für die Trainingsdaten
DTrain = zeros(nTrain,P); 
for n = 1:nTrain
   img = XTrain(:,:,1,n); 
   DTrain(n,:) = img(:); 
end
DTrain = [DTrain, ones(nTrain, 1)]; 
% Designmatrix für Testdaten
DTest = zeros(nTest,P); 
for n = 1:nTest
   img = XTest(:,:,1,n); 
   DTest(n,:) = img(:); 
end
DTest = [DTest, ones(nTest, 1)]; 
%% Normalengleichung lösen: so nicht! 
tic
wb = linsolve(DTrain' * DTrain, DTrain'*YTrain); 
toc


%% Jetzt mit Regularisierung
lambda = 1.5; 
tic
wb = linsolve(DTrain' * DTrain + lambda*eye(P+1), DTrain'*YTrain); 
toc
% Darstellung der Gewichte als Bild
w = wb(1:end-1);
imagesc(reshape(abs(w)./max(abs(w(:))), imSize));
colorbar; 

%% Vorhersage auf Trainingsdaten: 
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact');
nexttile(1); 
YTrainPred = DTrain * wb; 
scatter(YTrain, YTrainPred);
xlabel('Winkel: Gemessen'), ylabel('Winkel: Vorhersage'); 
RMSETrain = sqrt(mean( (YTrain - YTrainPred).^2 )); 
title('Trainingsdaten', sprintf('RMSE=%.2f, \\lambda=%.2f', RMSETrain, lambda)); 

%% Vorhersage auf Testdaten:
YTestPred = DTest * wb; 
nexttile(2);
scatter(YTest, YTestPred); 
xlabel('Winkel: Gemessen'), ylabel('Winkel: Vorhersage'); 
RMSETest = sqrt(mean( (YTest - YTestPred).^2 ));
title('Testdaten', sprintf('RMSE=%.2f, \\lambda=%.2f', RMSETest, lambda)); 

%% Ein paar Testbilder ansehen
idx = randperm(nTrain,20);
t = tiledlayout(4,5, 'TileSpacing','compact', 'Padding', 'compact'); 
for i = 1:numel(idx)
    nexttile;
    imshow(XTest(:,:,:,idx(i)));
    title(sprintf('%i° / %i°', YTest(idx(i)), round(YTestPred(idx(i)))));
end
title(t, 'Ein paar Testbilder', 'Echter Winkel / Vorhersagewinkel');

%% Loop über lambda
lambdaVec = logspace(-2, 2, 100); 
RMSETrainVec = zeros(size(lambdaVec));
RMSETestVec = zeros(size(lambdaVec));
tic
for ll = 1:length(lambdaVec)
   lambda = lambdaVec(ll); 
   wb = linsolve(DTrain' * DTrain + lambda*eye(P+1), DTrain'*YTrain); 
   
   % Trainingsdaten
   YTrainPred = DTrain * wb; 
   RMSETrainVec(ll) = sqrt(1/nTrain * sum( (YTrain - YTrainPred).^2 ));

   % Testdaten
   YTestPred = DTest * wb; 
   RMSETestVec(ll) = sqrt(1/nTest * sum( (YTest - YTestPred).^2 ));
end
toc

%% Graphische Darstellung
[minRMSE, idx] = min(RMSETestVec); 
t = tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact'); 
title(t, 'RMSE in Abhängigkeit von \lambda', ...
   sprintf('Bestes Test-RMSE: %.2f° bei \\lambda=%.2f', minRMSE, lambdaVec(idx))); 
nexttile(1);
plot(lambdaVec, RMSETrainVec, lambdaVec, RMSETestVec); 
title('Lineare \lambda-Achse');
xlabel('\lambda'), ylabel('RMSE (Grad)'); 
legend('Training', 'Test', 'Location', 'SE'); 
nexttile(2); 
semilogx(lambdaVec, RMSETrainVec, lambdaVec, RMSETestVec); 
xlabel('\lambda'), ylabel('RMSE (Grad)'); 
title('Logarithmische \lambda-Achse');