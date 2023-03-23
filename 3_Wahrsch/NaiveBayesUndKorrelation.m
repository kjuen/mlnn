%% Wie macht sich Naive Bayes bei Korrelation der Merkmale?
% Wenn hier x(:,1) und x(:,2) unkorreliert sind, klappt Naive Bayes gut. 
% Wenn man in der Kovarianzmatrix aber Korrelation einschaltet, klappt es
% nicht mehr gut!

N = 400; 
mu = [0; 0]; 
% C = [1/2, 3/4; 3/4, 2.5/2]; 
od = 0.5; 
C = [1/2, od; od, 2.5/2]; 
rhotheo = C(1,2) / sqrt(C(1,1)*C(2,2)); 
x = mvnrnd(mu,C,N);
corrMat = corrcoef(x); 
% Teile Klassen mit Ausgleichsgrade ein
p = polyfit(x(:,1), x(:,2), 1); 
lbl = x(:,2) > p(2) + p(1)*x(:,1);
gscatter(x(:,1), x(:,2), lbl);
%%
nb = fitcnb(x, lbl); 
lblprednb = predict(nb, x); 
trainErrnb = sum(lbl~=lblprednb)/N; 
knn = fitcknn(x, lbl, 'NumNeighbors',8); 
lblpredknn = predict(knn, x); 
trainErrknn = sum(lbl~=lblpredknn)/N; 
title({sprintf('Korr = %.2f', corrMat(1,2)); ...
   sprintf('Fehler (NB): %.1f%%, Fehler (kNN): %.1f%%', 100*trainErrnb, 100*trainErrknn)});

