%% Uebungen zu Kapitel 5

%% 1. X-Entropie des Muenzwurfs

N = 7;
NK = 4;
% Daten:
xs = @(t) -NK*log(t) - (N-NK)*log(1-t);

th = 0.001:0.001:0.99;
plot(th, xs(th));
hold on;
xline(NK/N, 'k--', 'LineWidth', 2);
plot(NK/N, xs(NK/N), 'xr', 'MarkerSize', 10);
xlabel('\theta', 'FontSize', 16), ylabel('XS(\theta)'); 
title(sprintf('Kreuzentropie %ixK, %ixZ', NK, N-NK));
legend('XS', sprintf('%i/%i', NK,N)); 

%% Anfangspunkt des Gradientenabstiegs
t0 = 0.1;
plot(t0, xs(t0), 'xg', 'DisplayName','\theta_0');
hold off; 

%% Anwenden des einfachen Gradientenabstiegs
nIts = 5;
eta = 0.01;
dxs = @(t) -NK/t + (N-NK)/(1-t);

[topt, track] = gaEinfach(dxs, eta, t0, nIts); 
xsVec = zeros(nIts, 1);
for n = 1:nIts
   xsVec(n) = xs(track(n));
end
hold on;
plot(track, xsVec, 'om', 'DisplayName', 'GA');
hold off;
