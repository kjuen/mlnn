%% Einfaches Beispiel für Gradientenabstieg

%% 1. Ein schmales Tal
a = 0.1; 
E2D = @(w) w(1)^2 + a*w(2)^2; 

[X,Y] = meshgrid(linspace(-3, 3,50));
Z = zeros(numel(X), 1);
for n = 1:numel(X)
   Z(n) = E2D([X(n), Y(n)]);
end
Z = reshape(Z, size(X));
f1 = figure();
surf(X,Y,Z);
shading interp;  light, lighting gouraud, material dull
alpha(0.8)
hold on;
contour3(X,Y,Z, 20, 'k:');
contour(X,Y,Z, 20);
hold off;

%%  2. einfacher Gradientenabstieg
gradE2D = @(w) [2*w(1); 2*a*w(2)];  

eta = 0.8;
w0 = [1.5; -1.5]; 
nIts = 10; 
[w, track] = gaEinfach(gradE2D, eta, w0, nIts);
Eeinfach = zeros(nIts+1, 1);
Eeinfach(1) = E2D(w0);
for n = 1:nIts
   Eeinfach(n+1) = E2D(track(:,n));
end
hold on; 
plot(w0(1), w0(2), 'rx', w(1), w(2), 'ro');
plot([w0(1), track(1,:)], [w0(2), track(2,:)],'r-');
plot3([w0(1), track(1,:)], [w0(2), track(2,:)], Eeinfach, 'r-');

hold off;

%% 5.  Verlauf der Energie- (oder Kosten-) funktion
f2 = figure;
plot(0:nIts, Eeinfach);
xlabel('Iteration'), ylabel('Energie'), 
title('Verlauf der Energiefunktion');

%% 3. NAG-Momentum-Methode
gamma = 0.9;
eta = 0.6;
[w, track] = gaNAG(gradE2D, eta, w0, nIts, gamma);
Enag = zeros(nIts+1, 1);
Enag(1) = E2D(w0);
for n = 1:nIts
   Enag(n+1) = E2D(track(:,n));
end
figure(f1);
hold on; 
plot(w0(1), w0(2), 'cx', w(1), w(2), 'co');
plot([w0(1), track(1,:)], [w0(2), track(2,:)],'c-');
plot3([w0(1), track(1,:)], [w0(2), track(2,:)], Enag, 'c-');
hold off;

%% 4. RMS-Prop
beta = 0.9;
eta = 0.6;
[w, track] = gaRmsProp(gradE2D, eta, w0, nIts, beta);
Ermsp = zeros(nIts+1, 1);
Ermsp(1) = E2D(w0);
for n = 1:nIts
   Ermsp(n+1) = E2D(track(:,n));
end
figure(f1);
hold on; 
plot(w0(1), w0(2), 'mx', w(1), w(2), 'mo');
plot([w0(1), track(1,:)], [w0(2), track(2,:)],'m-');
plot3([w0(1), track(1,:)], [w0(2), track(2,:)], Ermsp, 'm-');
hold off;

%% 5. Vergleich: Verlauf der Kostenfunktion
figure;
plot(0:nIts, Eeinfach, 0:nIts, Enag, 0:nIts, Ermsp);
xlabel('Iteration'), ylabel('Energie'), 
legend('Einfach', 'Nag-Mom', 'Rms-Prop'); 
title('Verlauf der Energiefunktion');
