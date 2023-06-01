function h = plotTrainingProgress(info, errLim)
   %PLOTTRAININGPROGRESS Eigener Plot des Training-Fortschritts

   if nargin <2
      errLim = 100; 
   end
   nIts = length(info.TrainingAccuracy); 
   
   plot(1:nIts, 100-info.TrainingAccuracy); 
   hold on;
   plot(1:nIts, 100-info.ValidationAccuracy, 'o');
   hold off;
   ylim([0, errLim]);
   legend('Trainingsfehler', 'Validierungsfehler'); 
   xlabel('Iterationen');
   ylabel('Fehler in %');
   title(sprintf('%i Iterationen', nIts));
end

