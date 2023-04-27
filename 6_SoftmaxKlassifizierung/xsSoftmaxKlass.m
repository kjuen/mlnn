function [xs, xsGrad, xsGradBatch] = xsSoftmaxKlass(xMat, lbl, lambda)
   %xsSoftmaxReg Kreuzentropie und Gradient der Softmax-Regression
   % Argumente
   % xMat: NxP - Datenmatrix
   % yMat: NxC one-hot-codierte Indikatormatrix der Klassenlabel
   % lambda (default=0): l2-Regularisierung
   
   if nargin==2
      lambda = 0;
   end

   assert(iscolumn(lbl));
   yMat = onehotencode(lbl, 2);
   
   C = size(yMat, 2);
   N = size(xMat, 1);
   assert(size(yMat, 1) == N);
   P = size(xMat, 2);
   
   function xs = xsImpl(wbVec)
      % xsFuncImpl: Kreuzentropie der Softmax-Regression

      % Gewichtsmatrix und Offset-Vektor extrahieren:
      [wMat, bVec] = reshapeWbVec(wbVec);
                
      zMat = xMat * wMat + bVec;   % Gl (6.2.5) im Skript
      assert(all(size(zMat) == [N, C]));
      
      % Berechnung der Kreuzentropie nach Gl (6.3.1) im Skript
      % ist numerisch nicht robust
      % pMat = sm(zMat);
      % xs = -sum(yMat .* log(pMat), 'all');
      
      % Besser: den Log erst vereinfachen
      zMat = zMat-max(zMat, [], 2);
      logSa = zMat - log(sum(exp(zMat), 2));   % Summe entlang der 2. Dim der Matrix
      xs = -sum(yMat .* logSa, 'all');
      assert(all(size(logSa) == [N, C]));
      
      % Regularisierung:
      xs = xs + lambda/2 * sum(wbVec.^2); 
   end
   xs = @xsImpl;   % Rückgabewert=Handle auf diese Funktion
   
   
   function xsGrad = xsGradImpl(wbVec)
      % xsGradImpl: Gradient der Kreuzentropie der Softmax-Regression
      
      % Gewichtsmatrix und Offset-Vektor extrahieren:
      [wMat, bVec] = reshapeWbVec(wbVec);
      
      zMat = xMat * wMat + bVec;   % Gl (6.2.5) im Skript
      assert(all(size(zMat) == [N, C]));
      pMat = sm(zMat);

      Delta = pMat - yMat;    % Gl (6.3.4) im Skript

      % Berechnung der Gradienten mit Gl (6.3.5)
      dXSdw = xMat' * Delta; 
      dXSdb = sum(Delta);       % Spaltensumme = Summe über Zeilenindizes
      
      % Flach machen als Spaltenvektor und Regularisierungsanteil zufügen:
      xsGrad = [dXSdw(:); dXSdb'] + lambda*wbVec; 
   end
   xsGrad = @xsGradImpl;
   
   function xsGrad = xsGradBatchImpl(wbVec, mask)
      % xsGradImpl: Gradient der Kreuzentropie der Softmax-Regression
      
      nBatch = sum(mask);
      % Gewichtsmatrix und Offset-Vektor extrahieren:
      [wMat, bVec] = reshapeWbVec(wbVec);
      
      xBatch = xMat(mask, :);
      yBatch = yMat(mask, :); 

      zMat = xBatch * wMat + bVec;   % Gl (6.2.5) im Skript
      assert(all(size(zMat) == [nBatch, C]));
      pMat = sm(zMat);

      Delta = pMat - yBatch;    % Gl (6.3.4) im Skript

      % Berechnung der Gradienten mit Gl (6.3.5)
      dXSdw = xBatch' * Delta; 
      dXSdb = sum(Delta);       % Spaltensumme = Summe über Zeilenindizes
      
      % Flach machen als Spaltenvektor und Regularisierungsanteil zufügen:
      xsGrad = [dXSdw(:); dXSdb'] + lambda*wbVec; 
   end
   xsGradBatch = @xsGradBatchImpl;

   % Hilfsfunktion, wird nicht exportiert
   function [wMat, bVec] = reshapeWbVec(wbVec)
      assert(length(wbVec) == P*C + C);
      % Den 1-D Vektor bw in Gewichtsmatrix und Offset-Vektor zerlegen:
      % Die ersten PxC Elemente bilden die Gewichtsmatrix
      wMat = reshape(wbVec(1:(P*C)), P, C);
      % Die hinteren C Elemente sind der Offset-Zeilenvektor
      bVec = reshape(wbVec((C*P+1):end), 1, C);
   end


end




