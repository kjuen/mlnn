%% Kreuzentropie Muenzwurf
clear, close;

% Daten: 4xK und 3xZ
xs = @(t) -4*log(t) - 3*log(1-t);

th = 0.001:0.001:0.99;
plot(th, xs(th)); 
hold on; 
xline(4/7, 'k--', 'LineWidth', 2);
plot(4/7, xs(4/7), 'xr', 'MarkerSize', 10); 
xlabel('\theta', 'FontSize', 16), ylabel('XS(\theta)', 'FontSize', 16); 
title('Kreuzentropie 4xK, 3xZ', 'FontSize', 16); 
legend('XS', '4/7', 'FontSize', 16); 

hold off; 
% ylim([4.5,5]); 