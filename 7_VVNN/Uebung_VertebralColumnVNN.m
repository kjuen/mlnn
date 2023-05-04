%% Vertebral Column (=Wirbelsäule) Datensatz: Naive-Bayes-Analyse
% https://archive.ics.uci.edu/ml/datasets/Vertebral+Column
% Zielgroessen: Orthopaedische Klassen: DH (Disk Hernia), Spondylolisthesis (SL), Normal (NO) and Abnormal (AB)
% Der Datensatz 2C enthaelt die Klassen AB und NO und der andere die
% Klassen DH, SL und NO. 
% 6 biomechanische Messwerte als Merkmale (das letzte ist die Klasse)

%% Laden und Normieren der Daten
fileName = fullfile('..', 'Datensaetze', 'VertebralColumn.csv'); 
T = readtable(fileName);
disp(head(T));

D = table2array(T(:,1:6));
lbl = categorical(T.orthopaedicClass);
tabulate(lbl);

m = mean(D); 
sigma = std(D); 
D = (D-m)./sigma;


%% Netzwerk mit einer inneren Schicht 
[N, P] = size(D); 
H = 5;  
C = 3; 

layers = [featureInputLayer(P)
   fullyConnectedLayer(H)
   reluLayer
   fullyConnectedLayer(C)
   softmaxLayer
   classificationLayer];

lambda = 0; 
options = trainingOptions('rmsprop',...
   'ValidationData', {D, lbl}, ...
   'MaxEpochs', 1000, ...
   'MinibatchSize', N, ...
   'L2Regularization', lambda, ...
   'Verbose', true, ...
   'Plots', 'none'); % training-progress'); 
% Training: 
tic
net = trainNetwork(D, lbl, layers, options);
toc
lblPred = classify(net, D);
err1 = sum(lblPred ~= lbl) / length(lbl); 


%% Zwei innere Schichten
[N, P] = size(D); 
H2 = 20;
H1 = 10;
C = 3; 
