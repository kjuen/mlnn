%% Uebungen zu Kapitel 5

%% 2. 2D-Funktion 
E = @(w) (w(1)-1).^2 .* (w(2)+1).^2; 

[X,Y] = meshgrid(linspace(-1, 2, 50), ...
   linspace(-2, 1, 50));
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
xlabel('w1'), ylabel('w2'), title('E(w1, w2)');
zlim([-1,4]);

%% Anfangspunkt des Gradientenabstiegs
w0 = [0;0];
plot3(w0(1), w0(2), E(w0), 'xg');
hold off; 

%% Anwenden Gradientenabstieg
eta = 0.3; 
nIts = 20; 
gradE = @(w) [2*(w(1)-1) .* (w(2)+1).^2; 
   2*(w(1)-1).^2 .* (w(2)+1)];
[wopt, track] = gaEinfach(gradE, eta, w0, nIts);
Evec = zeros(nIts, 1);
for n = 1:nIts
   Evec(n) = E(track(:,n));
end
hold on;
plot3(track(1,:), track(2,:), Evec, 'og');
hold off;