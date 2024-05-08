function [f] = fesfera(x,np)
% Función de repulsión para (n/3) puntos en la esfera unitaria de 
% dimensión tres.
%
% Optimización Numérica 
% ITAM
% 20 de octubre de 2020
%
n = length(x); % número de puntos en la esfera
f = 0;           % valor inicial de f
np=21;
for i = 1:np-1
    ui = x(3*(i-1)+1:3*i);         % anclados en el punto ui
   for j =i+1:np
        uj = x(3*(j-1)+1:3*j);      % punto uj
      f = f+(1/norm(ui-uj));    % la función crece en sumandos.
   end    
end

end
