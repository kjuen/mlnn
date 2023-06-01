function [w, track] = gaRmsProp(gradFunc, eta, w0, nIts, beta)
    %GAEINFACH Sehr einfache Version des Gradientenabstiegs 
    w = w0; 
    s = zeros(size(w0));
    SMALL = 1e-8;
    track = zeros(length(w0),nIts);
    for n = 1:nIts
       g = gradFunc(w);
       s = beta * s + (1-beta)* g.* g;
       etaVec = eta ./ sqrt(s+SMALL);  
       w = w - etaVec .* g;
       track(:,n) = w;
    end
end

