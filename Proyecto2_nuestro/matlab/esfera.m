%Script que utiliza los codigos pcsglobal, fesfera y hesfera para calcular
%puntos dentro de la esfera unitaria de cierta forma que cumplan con la
%función fesfera con restricciones hesfera y después dibuja los puntso en
%la esfera.
%pcsglobal: método para calcular el mínimo de la función fesfera con
%restricciones hesfera
%fesfera: función a minimizar
%hesfera: función que tiene las restricciones del problema. 
clear all; close all;
np=21; %número de puntos a generar
u=[1,0,0]'; %punto inicial
x=[u;rand(60,1)]; %llenado de los demás puntos utilizando números aleatorios
f=@(x)fesfera(x,np); %definir función y cuantos puntos
h=@(x)hesfera(x,np); %definir restriciones y cuantos puntos
tic
[y,L,k]=pcsglobal(f,h,x); %utilizar el método para calcular los puntos.
toc
sphere(50) %dibujamos la esfera
axis equal
hold on
h=zeros(21,1);
P1=zeros(3,21);
for j = 1:np %dibujar los puntos en la esfera
    P=y(3*(j-1)+1:3*j);
    P1(1:3,j)=P;
    plot3(P(1), P(2), P(3),'rd','Linewidth',3)
    hold on
    h(j)=P'*P;
end

%Genera la Tabla de los Puntos Generados
fprintf('Puntos Generados:')
fprintf('\n\t\t\t%s \t\t\t\t\t%s \t\t\t\t\t%s\n' ,'x','y','z');
fprintf('\t-------------------------------------------------------------------------------------------------');

fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,1),P1(2,1),P1(3,1));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,2),P1(2,2),P1(3,2));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,3),P1(2,3),P1(3,3));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,4),P1(2,4),P1(3,4));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,5),P1(2,5),P1(3,5));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,6),P1(2,6),P1(3,6));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,7),P1(2,7),P1(3,7));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,8),P1(2,8),P1(3,8));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,9),P1(2,9),P1(3,9));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,10),P1(2,10),P1(3,10));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,11),P1(2,11),P1(3,11));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,12),P1(2,12),P1(3,12));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,13),P1(2,13),P1(3,13));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,14),P1(2,14),P1(3,14));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,15),P1(2,15),P1(3,15));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,16),P1(2,16),P1(3,16));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,17),P1(2,17),P1(3,17));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,18),P1(2,18),P1(3,18));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,19),P1(2,19),P1(3,19));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,20),P1(2,20),P1(3,20));
fprintf('\n\t%s \t\t%d \t\t%.4d', P1(1,21),P1(2,21),P1(3,21));
fprintf('\n');
fprintf('\t-------------------------------------------------------------------------------------------------');
fprintf('\n');


