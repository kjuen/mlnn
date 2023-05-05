function p = sm(z)
   %SM Implementierung der Softmax function
   % sm arbeitet auf den Zeilen einer Matrix!
   % Zur Verbesserung der numerischen Stabilitaet muss man große positive 
   % Werte im Exponenten vermeiden. Daher zieht man das Maximum von z ab. 
   % Der Funktionswert bleibt daduch unveraendert. Fuer weitere Details siehe
   % https://nhigham.com/2021/01/12/what-is-the-softmax-function/
   
   z = z-max(z, [], 2);    % Max der Zeilen = Spaltenvektor
   p = exp(z);
   p = p ./ sum(p, 2);     % Teile durch die Zeilensummen
   
end

