%% Anpassen von Verteilungen
clear, close;

%% Normalverteilung
mu = 0.56; 
sigma = 1.67; 
N = 1000;
x = mu + sigma * randn(N,1); 
yyaxis left; 
h = histogram(x, 20); 
xl = get(gca, 'XLim'); 
muEst = mean(x); 
sigmaEst = std(x); 
varEst = var(x); 
xx = linspace(xl(1), xl(2), 200); 
yyaxis right; 
plot(xx, 1/sqrt(2*pi*varEst) * exp(-(xx-muEst).^2/(2*varEst))); 
[kDensVals, ~, bw] = ksdensity(x, xx); 
hold on; 
plot(xx, kDensVals); 
hold off; 
legend('Daten', 'Normalvert.', 'Kernel-Schätzung'); 


%% Verteilung mit 2 Peaks
mu1 = 2.56;
mu2 = -mu1; 
sigma1 = 1.67;
sigma2 = 0.78; 
N = 1000;
x = [mu1 + sigma1 * randn(N/2,1); ...
   mu2 + sigma2 * randn(N/2,1)];
yyaxis left; 
h = histogram(x, 20); 
xl = get(gca, 'XLim'); 
muEst = mean(x); 
sigmaEst = std(x); 
varEst = var(x); 
xx = linspace(xl(1), xl(2), 500); 
yyaxis right; 
plot(xx, 1/sqrt(2*pi*varEst) * exp(-(xx-muEst).^2/(2*varEst))); 

[kDensVals, ~, bw] = ksdensity(x, xx, 'Bandwidth', 0.8); 
hold on; 
plot(xx, kDensVals, 'm-'); 
hold off; 
legend('Daten', 'Normalvert.', 'Kernel-Schätzung'); 



