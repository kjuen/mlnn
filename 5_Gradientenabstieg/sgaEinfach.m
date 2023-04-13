function [w, track] = sgaEinfach(gradFunc, eta, w0, N, nEpochs, mbSize)
    %SGAEINFACH Sehr einfache Version des stochastischen Gradientenabstiegs 
    % Argumente:
    % gradFunc: Ableitungsfunktion der Funktion, Spaltenvektor mit P Komponenten, 
    %           muss zwei Argumente akzeptieren: den w-Vektor und die
    %           Minibatch-Indizes!
    % eta, w0: wie bei deterministischem Gradientenabstieg
    % N: Groesse des Datensatzes
    % nEpochs: Anzahl der Epochen
    % mbSize: Groesse des Minibatches 

    % Minibatch-Management (gleiche Logik wie bei Kreuzvalidierung)
    nMbs = floor(N / mbSize);   % Anzahl minibatches
    rest = N - nMbs * mbSize;
    mbIdx = [repmat(1:nMbs, 1, mbSize), 1:rest];

    w = w0;
    track = zeros(length(w0), nEpochs * nMbs);
    it = 1; % Laufindex, der die Iterationen zaehlt
    for ep=1:nEpochs  % Schleife über die Epochen
       % Wuerfele Indizes 1:N durcheinander, damit jede Epoche anders ist:
       shuffledIdx = mbIdx(randperm(N));
       
       % Schleife ueber die Minibatches
       for mb = 1:nMbs
          mbMask = shuffledIdx==mb;  % definiert aktuellen Minibatch
          mbGrad = gradFunc(w, mbMask);
          w = w - eta * mbGrad;
          track(:,it) = w; 
          it = 1 + it;
       end
    end
end

