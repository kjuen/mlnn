function [MSEFunc, MSEGradFunc, MSEGradFuncBatch] = createMSEFuncGerade(x,y)
   %createMSEFuncSchwingung: MSE-Funktion und Gradient einer
   %Regressionsgerade
   
   % Programmierprinzip: 'Closure'
   % erzeugte Funktionen haben Zugriff  auf ihren
   % Erstellungskontext, hier also auf die Daten x,y
   % siehe z.B. https://de.wikipedia.org/wiki/Closure_(Funktion)
   
   N = length(x); 
   
   function MSE = MSEImpl(wb)
      w = wb(1);
      b = wb(2); 
      z = w*x + b; 
      MSE = 1/(2*N) * sum( (z-y).^2);
   end
   MSEFunc = @MSEImpl; 
   
   function MSEGrad = MSEGradImpl(wb)
      w = wb(1);
      b = wb(2);
      z = w*x + b;
      Delta = z-y;
      MSEGrad = [1/N*Delta'*x; mean(Delta)]; 
   end
   MSEGradFunc = @MSEGradImpl;
   

   function MSEGradBatch = MSEGradImplBatch(bw, mask) 
      
      b = bw(1);
      w = bw(2); 
      
      xBatch = x(mask);
      yBatch = y(mask); 
      
      z = b + w*xBatch;
      Delta = 2*(z-yBatch); 
      MSEGradBatch = [mean(Delta); mean(xBatch.*Delta)]; 
   end
   MSEGradFuncBatch = @MSEGradImplBatch; 
end

