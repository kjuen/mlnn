%% Uebung 1 zu Kapitel 5: X-Entropie des Muenzwurfs

N = 7;
NK = 4;
% Daten:
xs = @(t) -NK*log(t) - (N-NK)*log(1-t);

th = 0.001:0.001:0.99;
plot(th, xs(th));
hold on;
xline(NK/N, 'k--', 'LineWidth', 2);
plot(NK/N, xs(NK/N), 'xr', 'MarkerSize', 10);
xlabel('\theta', 'FontSize', 16), ylabel('XS(\theta)', 'FontSize', 16);
title(sprintf('Kreuzentropie %ixK, %ixZ', NK, N-NK), 'FontSize', 16);
legend('XS', sprintf('%i/%i', NK,N), 'FontSize', 16);

%% Anfangspunkt des Gradientenabstiegs
t0 = 0.1;
plot(t0, xs(t0), 'xg', 'DisplayName','\theta_0');
hold off; 

%% Anwenden des einfachen Gradientenabstiegs
nIts = 5;


