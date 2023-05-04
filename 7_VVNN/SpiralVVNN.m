%% Spiral-Daten
load(fullfile('..', 'Datensaetze', 'spiral'));
xdim = [min(xySpiral(:,1)), max(xySpiral(:,1))];
ydim = [min(xySpiral(:,2)), max(xySpiral(:,2))];
gscatter(xySpiral(:,1), xySpiral(:,2), lblSpiral, 'rg', '..', 15*[1,1]);
xlim(xdim), ylim(ydim);
lblSpiral = categorical(lblSpiral);
[N, P] = size(xySpiral);
C = 2;

%% Ein breites, flaches Netz
H = 200;
block = [fullyConnectedLayer(H)
   reluLayer
   batchNormalizationLayer];
layers = [featureInputLayer(P)
   block
   block
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];

% Training-Optionen
lambda = 0.0;
options = trainingOptions('rmsprop',...
   'ValidationData', {xySpiral, lblSpiral}, ...
   'MaxEpochs', 2000, ...
   'MinibatchSize', N, ...
   'L2Regularization', lambda, ...
   'OutputNetwork', 'best-validation-loss', ...
   'Verbose', true, ...
   'Plots', 'None');

tic
net = trainNetwork(xySpiral, lblSpiral, layers, options);
toc
lblPred = classify(net, xySpiral);
errTrain = mean(lblPred ~= lblSpiral);
fprintf('Fehler = %.2f%%, #Parameter = %i\n', 100*errTrain, nParamsVVNN(net));
%% Klassifizierungsregionen
nGrid = 100;
[Xg,Yg] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
   linspace(ydim(1), ydim(2), nGrid));
Dgrid = [Xg(:), Yg(:)];
[~, pGrid]  = classify(net, Dgrid);
pGrid = reshape(pGrid,[size(Xg), 2]);
img = zeros(nGrid, nGrid, 3);
img(:,:,1:2) = pGrid;
% Eintragen in den Scatterplot:
gscatter(xySpiral(:,1), xySpiral(:,2), lblSpiral, 'rg', '..', 15*[1,1]);
hold on;
image(img, 'XData', xdim, 'YData', ydim);
% Trennlinien
contour(Xg,Yg,pGrid(:,:,1), 0.5*[1,1], 'r', 'Linewidth', 2, 'DisplayName', 'Trennl 1');
contour(Xg,Yg,pGrid(:,:,2), 0.5*[1,1], 'g', 'Linewidth', 2, 'DisplayName', 'Trennl 2');
alpha(0.3);  % Transparenz des Hintergrundes
hold off;
legend off;
axis([xdim, ydim]);
title('Trainingsdaten', ...
   sprintf('Klassifizierungsregionen mit \\lambda = %.3f', lambda));


%% Ein tiefes, schmales Netz
H = 25;
block = [fullyConnectedLayer(H)
   reluLayer
   batchNormalizationLayer];

layers = [featureInputLayer(P)
   block
   block
   block
   block
   block
   block
   block
   block
   block
   block
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];


%% 
lambda =  0.0; % 0.25;
options = trainingOptions('rmsprop',...
   'ValidationData', {xySpiral, lblSpiral}, ...
   'MaxEpochs', 2000, ...
   'MinibatchSize', N, ...
   'L2Regularization', lambda, ...
   'OutputNetwork', 'best-validation-loss', ...
   'Verbose', true, ...
   'Plots', 'None');

tic
net = trainNetwork(xySpiral, lblSpiral, layers, options);
toc
lblPred = classify(net, xySpiral);
errTrain = mean(lblPred ~= lblSpiral);
fprintf('Fehler = %.2f%%, #Parameter = %i\n', 100*errTrain, nParamsVVNN(net));

%% Klassifizierungsregionen
nGrid = 100;
[Xg,Yg] = meshgrid(linspace(xdim(1), xdim(2), nGrid), ...
   linspace(ydim(1), ydim(2), nGrid));
Dgrid = [Xg(:), Yg(:)];
[~, pGrid]  = classify(net, Dgrid);
pGrid = reshape(pGrid,[size(Xg), 2]);
img = zeros(nGrid, nGrid, 3);
img(:,:,1:2) = pGrid;
% Eintragen in den Scatterplot:
gscatter(xySpiral(:,1), xySpiral(:,2), lblSpiral, 'rg', '..', 15*[1,1]);
hold on;
image(img, 'XData', xdim, 'YData', ydim);
% Trennlinien
contour(Xg,Yg,pGrid(:,:,1), 0.5*[1,1], 'r', 'Linewidth', 2, 'DisplayName', 'Trennl 1');
contour(Xg,Yg,pGrid(:,:,2), 0.5*[1,1], 'g', 'Linewidth', 2, 'DisplayName', 'Trennl 2');
alpha(0.3);  % Transparenz des Hintergrundes
hold off;
legend off;
axis([xdim, ydim]);
title('Trainingsdaten', ...
   sprintf('Klassifizierungsregionen mit \\lambda = %.3f', lambda));
