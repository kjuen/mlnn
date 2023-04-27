function [lbl, pMat] = applySoftmaxKlass(xMat, wMat, bVec)
   %APPLYSOFTMAXREG wende Softmax-Regression zur Vorhersage an
  
   [N,P] = size(xMat);
   C = length(bVec); 
   assert(isrow(bVec)); 
   assert(size(wMat,1) == P); 
   assert(size(wMat, 2) == C); 
   
   zMat = xMat * wMat + bVec;   % Gl (6.2.5) im Skript
   % Anwenden von sm auf die Zeilen von zMat, um eine 
   % NxC-Matrix mit Wahrscheinlichkeiten zu erhalten:
   pMat = sm(zMat);             
   assert(size(pMat,1)==N); 
   assert(size(pMat,2)==C);
   
   % Bestimme zeilenweise den Index des maximalen Wahrscheinlichkeit: das ergibt
   % das Klassenlabel: 
   [~, lbl] = max(pMat, [], 2);
   lbl = categorical(lbl);
end

