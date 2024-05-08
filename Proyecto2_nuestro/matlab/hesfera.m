function [h] = hesfera(x,np)
% Función de restricciones del problema de np puntos en la esfera unitaria
% de dimensión tres.
%
%  Optimzación Numérica
% ITAM
% 20 de octubre de 2020
%g =[];     % uso de fmincon.m en Matlab
np=21;
h = zeros(np,1);
for j = 1:np
    uj = x(3*(j-1)+1:3*j);
    h(j) = uj'*uj-1;
end
end