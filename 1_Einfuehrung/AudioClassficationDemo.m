%% Audio Classification Demo
% Based on Speech Command Recognition Using Deep Learning example in Audio
% Toolbox

%% Setup audio recorder and classifier

load('commandNet.mat');
Fs = 16000;

%% Start recording 
a = audiorecorder(Fs, 16, 1); 
a.record; 

%% Stop recording and extract max 1 second of speech
a.stop; 
x = a.getaudiodata;
N = length(x); 
t = (0:(N-1)) * 1/Fs; 
plot(t,x); 
xlabel('Zeit (Sek.)'); 
%% Retrieve audio signal and classify

idx = detectSpeech(x, Fs);
idx1 = min(idx(:));
idx2 = max(idx(:));
x2 = x(idx1:idx2);
if length(x2) > Fs
   error('Speech signal must not be longer than 1 Second'); 
end
soundsc(x, Fs); 
%% 
auditorySpect = helperExtractAuditoryFeatures(x2,Fs);
[command, scores] = classify(trainedNet,auditorySpect);
fprintf('Detected command: %s, score = %.2f\n', command, max(scores)); 

%% Die besten scores ausdrucken:
classNames = trainedNet.Layers(end).ClassNames;
[~, idx] = sort(scores, 'descend');
T = table(categorical(classNames(idx)), scores(idx)', 'VariableNames', {'Klasse', 'Wahrsch.'});
disp(head(T)); 


%% Print spectrogram
pcolor(auditorySpect);
xlabel('Zeit'); ylabel('Frequenz'); 

