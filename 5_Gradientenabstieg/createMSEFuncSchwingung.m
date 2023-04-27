function [MSEFunc, MSEGradFunc, MSEBatchGradFunc] = createMSEFuncSchwingung(t,y)
   %createMSEFuncSchwingung: MSE-Funktion und Gradient für Beispiel der
   %abklingenden Schwingung
   
   N = length(t);
   
   function MSE = MSEImpl(w)
      z = exp(w(1)*t) .* sin(w(2)*t);
      MSE = 1/(2*N)* sum( (z-y).^2);
   end
   MSEFunc = @MSEImpl; 
   
   function MSEGrad = MSEGradImpl(w) 
      z = exp(w(1)*t) .* sin(w(2)*t);
      Delta = (z-y);
      dzdw1 = t.*z; 
      dzdw2 = t.*exp(w(1)*t).*cos(w(2)*t);
      
      MSEGrad = [mean(Delta .* dzdw1); mean(Delta.*dzdw2)]; 
   end
   MSEGradFunc = @MSEGradImpl; 

   function MSEGrad = MSEBatchGradImpl(w, mbMask) 
      % Waehle die Daten des aktuellen Batches aus:
      xBatch = t(mbMask);
      yBatch = y(mbMask); 

      % ab hier die gleiche Rechnung wie in MSEGradImpl
      z = exp(w(1)*xBatch) .* sin(w(2)*xBatch);
      Delta = (z-yBatch);
      dzdw1 = xBatch.*z; 
      dzdw2 = xBatch.*exp(w(1)*xBatch).*cos(w(2)*xBatch);
      
      MSEGrad = [mean(Delta.*dzdw1); mean(Delta.*dzdw2)]; 
   end
   MSEBatchGradFunc = @MSEBatchGradImpl; 
end

