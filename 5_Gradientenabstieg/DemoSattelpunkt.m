%% Beispiel für einen Sattelpunkt

%% Monkey Saddle
E = @(w) w(1)^3 - 3*w(1)*w(2)^2; 
gradE = @(w) [3*w(1)^2 - 3*w(2)^2; -6*w(1)*w(2)]; 


[X,Y] = meshgrid(linspace(-4, 4,50));
Z = zeros(numel(X), 1);
for n = 1:numel(X)
   Z(n) = E([X(n), Y(n)]);
end
Z = reshape(Z, size(X));
surf(X,Y,Z);
shading interp;  light, lighting gouraud, material dull
alpha(0.8)
hold on;
contour3(X,Y,Z, 20, 'k:');
contour(X,Y,Z, 20);
hold off;
xlabel('w1'), ylabel('w2'), title('Monkey-Saddle');

%% Startwert fuer Gradientenabstieg
w0 = [0.1;0.1]; 
hold on; 
plot3(w0(1), w0(2), E(w0), 'xr');

%% Einfacher Gradientenabstieg
eta = 0.01; 
nIts = 200;
[w, track] = gaEinfach(gradE, eta, w0, nIts);
Eeinfach = zeros(nIts+1, 1);
Eeinfach(1) = E(w0);
for n = 1:nIts
   Eeinfach(n+1) = E(track(:,n));
end
plot3([w0(1), track(1,:)], [w0(2), track(2,:)], Eeinfach, 'r-');

%% RMS-Prop
eta = 0.01; 
nIts = 200;
beta = 0.9; 
[w, track] = gaRmsProp(gradE, eta, w0, nIts, beta);
Ermsp = zeros(nIts+1, 1);
Ermsp(1) = E(w0);
for n = 1:nIts
   Ermsp(n+1) = E(track(:,n));
end
plot3([w0(1), track(1,:)], [w0(2), track(2,:)], Ermsp, 'c-');
