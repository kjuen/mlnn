%% Die Faltungsoperation

%% Warum Faltung benoetigt wird

X = im2gray(imread('sherlock.jpg')); 
disp(size(X)); 
imshow(X); 

%%
% Bilder sind so groß, dass eine vollverbundene Schicht zu viel Speicher
% verbraucht.

% Initialisierung einer Gewichtsmatrix
P = numel(X); 
H = P/2; 
% W = zeros(P,H);    % -> Fehler: out of memory

%% Ein Bild als Vektor
% Speichert man ein Bild als Vektor, geht der räumliche 
% Zusammenhang zwischen den Pixel verloren.
img = imread('mnist_zero.png'); 
tiledlayout(2,1,'TileSpacing','compact', 'Padding', 'compact');  
nexttile
imshow(img); 
nexttile
imshow(repmat(img(:),200)'); 




%% Faltung in 1D
x = [1 2 -1 -1]; 
h = [1, -1]; 
y = conv(x,h); 

%% Das ganze als Matrixmultiplikation
r = [h zeros(1,length(x)-1)];
c = [h(1) zeros(1,length(x)-1)];
hConvMat = toeplitz(r,c);
y2 = hConvMat * x';

%% 1D-Filter zur Kantendetektion
x= linspace(-5,5,200); 
y = tanh(10*x); 
plot(x,y); 
h = [1, -1];
% h = [-1 8 0 -8 1]/12;  % 5-Punkt-Formel
% h = [1 -2 1]; % Laplace
kanten = conv(y,h, 'same'); 
hold on; 
plot(x,kanten);
hold off; 
axis([-4.5, 4.5, -1.25, 1.25]);
legend('Signal', 'Faltung', 'Location', 'NW'); 



%% X-Korrealation mit offset: dlconv
% dlconv implementiert nicht Faltung sondern Kreuzkorrelation (= Faltung
% ohne Filter-Flip). 
x = [1 2 -1 -1]; 
h = [1, -1]; 
dlx = dlarray(x', 'S'); 
% Unterscheidet sich um -1, da dlconv das Filter nicht 'flippt'
offset = 0; 
y = dlconv(dlx, h', offset, 'Padding', 1); 




%% Ein Beispiel mit Padding und Stride
x = [1 2 -1 -1]; 
h = [1, -1]; 
dlx = dlarray(x', 'S'); 
% Unterscheidet sich um -1, da dlconv das Filter nicht 'flippt'
offset = 0; 
y = dlconv(dlx, h', offset, 'Padding', 2, 'Stride', 1); 



%% X-Korr bei gleicher Länge und Padding = 0 ist das Skalarprodukt
x = [1; 2; -1; -1]; 
h = [1; 2; 3; 4];
y = dlconv(dlarray(x, 'S'), h, 0, 'Padding', 0)
x'*h



%% Bsp: Flankendetektion
b = -7.6; 
w = 10; 
x= linspace(-5,5,200)'; 
y = tanh(w*x+b); 
y = y + 0.02*randn(size(y));   % 0.02
% Linke y-Achse: Eingangssignal
yyaxis left; 
plot(x,y); 
title('Eingangssignal'); 
ylim([-1.1,1.1]); 
h = [1; -2; 1]; % Laplace
dly = dlarray(y, 'S'); 
bias = -0.15; 
dlkanten = dlconv(dly,h, bias, 'Padding', 0); 
% Rechte Achse: Ausgang nach Conv und relu
yyaxis right; 
plot(x(2:end-1), extractdata(relu(dlkanten))); 
hold on, yline(0, 'k--'), hold off; 
legend('Eingangssignal', 'Laplace-Filter + relu', 'Location', 'NW'); 


%% Pooling Beispiel
dla = dlarray([4 3 2 7 1 1 7 8 1 9 6 8], 'S');
poolSize =  4; 
maxpool(dla, poolSize, 'Padding',0, 'Stride',4)

%% Pooling-Beispiel: nochmal Flanken-Lokalisation

% oberer Plot genau wie eben
b = -17.6; 
w = 10; 
x= linspace(-5,5,200)'; 
y = tanh(w*x+b); 
y = y + 0.02*randn(size(y));   % 0.02
tiledlayout(2,1, 'TileSpacing','compact', 'Padding', 'compact'); 
nexttile; 
yyaxis left; 
plot(x,y); 
title('Eingangssignal'); 
ylim([-1.1,1.1]); 
h = [1; -2; 1]; % Laplace
dly = dlarray(y, 'S'); 
bias =  -0.15; 
dlkanten = dlconv(dly,h, bias, 'Padding', 0); 
yyaxis right; 
plot(x(2:end-1), extractdata(relu(dlkanten))); 
hold on, yline(0, 'k--'), hold off; 
legend('Eingangssignal', 'Laplace-Filter + relu', 'Location', 'NW'); 

% Unterer Plot: Max-Pooling
nexttile; 
dlp = maxpool(relu(dlkanten), 24, 'Padding', 0, 'Stride', 24); 
stem(1:length(dlp), extractdata(dlp), 'filled', 'Linewidth', 2);
title('Nach Pooling');


%% Faltung in 2D

X = [-3 0 1; 3 -3 1; 1 3 0]; 
h = [1 0; -1 2]; 
dlX = dlarray(X, 'SS'); 
dlY = dlconv(dlX, h, 0, 'Padding',  0, 'Stride', 1); 

%% Jetzt mit Bildern: zuerst blurring
% das könnte man genauso gut mit conv2 machen, wenn man kein Stride braucht! 

Xc = imread('sherlock.jpg');
X = single(im2gray(Xc));      % dlconv kommt mit uint8 nicht klar

% Definition des Blurring Filter als Durchschnitt ueber alle Pixel
len = 5; 
hAvg = 1/len^2 * ones(len); 

Y = extractdata(dlconv(dlarray(X, 'SS'), hAvg, 0));
montage( {uint8(X), uint8(Y)});


%% Kanten mit erster Ableitung
hPrewittX = [-1 0 1; -1 0 1; -1 0 1]; 
hPrewittY = [-1 -1 -1; 0 0 0; 1 1 1]; 

dlY = dlconv(dlarray(X, 'SS'), hAvg, 0);
Yx = conv2(X,hPrewittX);
Yy = conv2(X,hPrewittY);
Yx = extractdata(dlconv(dlarray(X, 'SS'), hPrewittX, 0));
Yy = extractdata(dlconv(dlarray(X, 'SS'), hPrewittY, 0));
% Problem: Faltung erzeugt Werte außerhalb von [0, 255] -> Skalieren auf
% richtigen unit8-Wertebereich mit mat2gray!
montage( {mat2gray(X), mat2gray(Yx), mat2gray(Yy)}, 'Size', [3,1]);

%% Laplace-Filter
% hLaplace = [1 1 1; 1 -8 1; 1 1 1]; 
hLog = fspecial('log', 9,3); 
[xg, yg] = meshgrid(1:9); surf(xg,yg,hLog); 
Yl = extractdata(dlconv(dlarray(X, 'SS'), hLog, 0));
montage( {mat2gray(X), mat2gray(Yx), mat2gray(Yy), mat2gray(Yl)}, 'Size', [2, 2]);


%% Der Mond

X = single(imread('moon.tif')); 
imshow(mat2gray(X));
%% Kantenfilter mit log-Filter
h = fspecial('log',15, 1); 
Y = extractdata(dlconv(dlarray(X, 'SS'), h, 0));
Y(Y<10) = 0;   % Rauschen entfernen

montage( {mat2gray(X), mat2gray(Y)});
%% Ganz ähnlich geht auch mit bias und relu

bias = -20; 
dlY = relu(dlconv(dlarray(X, 'SS'), h, bias));
% Max-Pooling vergroessert die Kanten
dlY = maxpool(dlY, 5);

Y = extractdata(dlY); 
montage( {mat2gray(X), mat2gray(Y)});


%% Faltung mit 3 Farbkanälen: % Ueber die Farbkanaele wird summiert!
% Reproduktion des Spielzeug-Beispiels: 
% https://medium.datadriveninvestor.com/convolutional-neural-networks-3b241a5da51e

X = zeros(5,5,3); 
X(:,:,1) = [156, 155, 156, 158, 158; 153, 154, 157, 159, 159; 149, 151, 155, 158, 159; ...
   146, 146, 149, 153, 158; 145, 143, 143, 148, 158];
X(:,:,2) = X(:,:,1) + [11;11;11;10;10]; 
X(:,:,3) = X(:,:,1) + [7;7;7;9;9];  

h = zeros(3,3,3); 
h(:,:,1) = [-1, -1, 1; 0, 1, -1; 0, 1, 1]; 
h(:,:,2) = [1, 0, 0; 1, -1, -1; 1, 0, -1];
h(:,:,3) = [0, 1, 1; 0, 1, 0; 1, -1, 1];
bias = 1; 

dlY = dlconv(dlarray(X, 'SSC'), h, bias, 'Padding', 1)



%% Ein richtiges Bild mit 3 Farbkanälen
X = imread('sherlock.jpg'); 
imshow(X); 
dlX = dlarray(single(X),'SSC');

%% Unterschiedliche Faltung für alle 3 Farbkanäle, am Ende wird summiert

hsize = 19;
h = zeros(hsize, hsize, 3); 
h1 = rand(hsize); 
h1 = h1/sum(h1(:)); 
h(:,:,1) = h1;  % blurring im roten Kanal
h(:,:,2) = fspecial('log', hsize);  % Kanten im gruenen Kanal
h(:,:,3) = fspecial('log', hsize);  % Kanten im blauen Kanal
dlY = dlconv(dlX,h,0,'Stride',1, 'Padding', 0); 
Y = extractdata(dlY);
montage( {X, uint8(Y)}, 'Size', [2,1]);

%% Auch ein 1x1x3-Faltungskern macht Sinn
h = zeros(1,1, 3); 
h(:,:,1) = 0;    
h(:,:,2) = 1;    % fische nur den grünen Kanal raus
h(:,:,3) = 0; 
dlY = dlconv(dlX,h,0,'Stride',1, 'Padding', 0); 
Y = extractdata(dlY);
montage( {X, X(:,:,2), uint8(Y)}, 'Size', [3,1]);

