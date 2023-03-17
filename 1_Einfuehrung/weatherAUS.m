%% Wetter in Australien
% Regnet es morgen oder nicht? 

filename = fullfile('..', 'Datensaetze', 'weatherAusTrain.xls');
opts = detectImportOptions(filename);
opts = setvartype(opts,{'Temp9am', 'Temp3pm', 'Pressure9am', 'Pressure3pm', 'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'},...
                        {'double', 'double',  'double',      'double',      'double',  'double',  'double',      'double',   'double',   'double'});
disp([opts.VariableNames', opts.VariableTypes']);
preview(filename, opts)
%% Daten laden
T = readtable(filename, opts); 

datMat = [T.MinTemp, T.MaxTemp, T.Rainfall, T.Evaporation, T.Sunshine, ...
   T.WindGustSpeed, T.WindSpeed9am, T.WindSpeed3pm, T.Humidity9am, T.Humidity3pm, ...
   T.Pressure3pm, T.Pressure9am, T.Cloud3pm, T.Cloud9am, T.Temp9am, T.Temp3pm];
locs = categorical(T.Location);
a1 = windDir2Angle(categorical(T.WindDir3pm));
a2 = windDir2Angle(categorical(T.WindDir9am)); 
a3 = windDir2Angle(categorical(T.WindGustDir)); 

trainDat = [datMat, grp2idx(locs), a1, a2, a3]; 
trainLbl = categorical(T.RainTomorrow); 




%% knn-Modell lernen
tic
Mdl = fitcknn(trainDat, trainLbl, 'NumNeighbors',40);
toc

%% Die Testdaten laden
filename = fullfile('..', 'Datensaetze', 'weatherAusTest.xls');
T = readtable(filename, opts); 

datMat = [T.MinTemp, T.MaxTemp, T.Rainfall, T.Evaporation, T.Sunshine, ...
   T.WindGustSpeed, T.WindSpeed9am, T.WindSpeed3pm, T.Humidity9am, T.Humidity3pm, ...
   T.Pressure3pm, T.Pressure9am, T.Cloud3pm, T.Cloud9am, T.Temp9am, T.Temp3pm];
locs = categorical(T.Location);
a1 = windDir2Angle(categorical(T.WindDir3pm));
a2 = windDir2Angle(categorical(T.WindDir9am)); 
a3 = windDir2Angle(categorical(T.WindGustDir)); 

testDat = [datMat, grp2idx(locs), a1, a2, a3]; 
%% Regnet es morgen beim ersten Datenpunkt? 
predict(Mdl, testDat(1,:))

