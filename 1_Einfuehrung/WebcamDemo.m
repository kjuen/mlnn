%% Webcam Demo
% webcamlist
% camera = webcam('/dev/video2');
% camera = webcam('Microsoft Camera Rear'); 
% camera = webcam('Logitech QuickCam Fusion');
camera = webcam('Logitech QuickCam Pro 9000'); 
net = googlenet;
inputSize = net.Layers(1).InputSize(1:2);

%%
figure
im = snapshot(camera);
image(im)
%%

im = imresize(im,inputSize);
[label,score] = classify(net,im);
title(sprintf('%s: (P = %.1f %%)', char(label), 100*max(score)));

%% 
h = figure;

while ishandle(h)
    im = snapshot(camera);
    image(im)
    im = imresize(im,inputSize);
    [label,score] = classify(net,im);
    title(sprintf('%s: (P = %.1f %%)', char(label), 100*max(score)));
    drawnow
end


%% Das Netz liefert Wahrscheinlichkeiten:
% Bild laden
classNames = net.Layers(end).ClassNames;
img = imread('sherlock.jpg');
img = imresize(img, inputSize); 
image(img);

%% Bild klassifizieren und Wahrscheinlichkeit angeben
[label,score] = classify(net,img);
title(sprintf('Klasse: %s, (P= %.3f)', char(label), max(score))); 

%% Klassen nach Wahrscheinlichkeiten sortieren und die besten Klassen auflisten:
[~, idx] = sort(score, 'descend');
T = table(categorical(classNames(idx)), score(idx)', 'VariableNames', {'Klasse', 'Wahrsch.'});
disp(head(T)); 




