function nParams = nParamsVVNN(net)
   %NPARAMSVVNN Berechnet die Anzahl der Parameter in einem
   %vollverbundenden Netz
  
   nParams = 0;
   for n = 1:length(net.Layers)
      layer = net.Layers(n);
      if isa(layer, 'nnet.cnn.layer.FullyConnectedLayer')
         nParams = nParams + numel(layer.Weights);
         nParams = nParams + numel(layer.Bias);
      end
   end


end

