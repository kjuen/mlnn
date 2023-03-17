function [testErr, trainErr] = xval(trainFunc, predictFunc, dat, lbl, ...
        nChunks, nRuns)
    %XVAL Generic cross validation function
    % Input: 
    % * Trainingsfunktion L = trainfunc(dat, lbl)
    % * Vorhersage lbl = predictFunc(L, dat)
    % * dat: Datenmatrix, NxP
    % * lbl: Spaltenvektor mit Labeln, Nx1
    % * nChunks (default=3): in wieviel Teile wird der Datensatz aufgeteilt
    % * nRuns (default=10): wie oft das ganze
    
    N = size(dat, 1);
    if nargin <= 4
        nChunks = 3;
    end
    if nargin <= 5
        nRuns = 10;
    end
    
    chunkSize = floor(N/nChunks);
    rest = N - nChunks*chunkSize;
    chunkIdx = [repmat(1:nChunks, 1, chunkSize), 1:rest];
    
    testErr = zeros(nRuns*nChunks, 1);
    trainErr = zeros(nRuns*nChunks, 1);
    count = 0;
    for r=1:nRuns
       idx = chunkIdx(randperm(N));
        for k=1:nChunks
            count = count + 1;
            % Mit idx==k wird getestet, mit den anderen trainiert
       
            testMask = (idx == k); 
            trainMask = ~testMask; 
            
            state = trainFunc(dat(trainMask, :), lbl(trainMask,:));
            testLbl = predictFunc(state, dat(testMask, :));
            testErr(count) = sum(testLbl ~= lbl(testMask))/sum(testMask);
            trainLbl = predictFunc(state, dat(trainMask, :));
            trainErr(count) = sum(trainLbl ~= lbl(trainMask))/sum(trainMask);
        end
    end
end

