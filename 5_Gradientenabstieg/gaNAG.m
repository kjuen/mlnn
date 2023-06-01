function [w, track] = gaNAG(gradFunc, eta, w0, nIts, gamma)
    %GANAG Sehr einfache Version des Nesterov-Gradientenabstiegs mit Momentum
    
    w = w0; 
    track = zeros(length(w0),nIts);
    v = zeros(size(w0));
    for n = 1:nIts
       v = gamma * v - eta * gradFunc(w+gamma*v);
       w = w + v;
       track(:,n) = w;
    end
end

