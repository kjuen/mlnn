function [w, track] = gaEinfach(gradFunc, eta, w0, nIts)
    %GAEINFACH Sehr einfache Version des Gradientenabstiegs 
    w = w0; 
    track = zeros(length(w0),nIts);
    for n = 1:nIts
        w = w - eta * gradFunc(w);
        track(:,n) = w; 
    end
end

