function [x, lambda]= pc(Q,A,c,b)
%metodo directo para resolver el problema 
%min (1/2)*x'Qx+c'*x
%SA A*x=b
%Q es una matriz spd 
%A es de mxn con rango(A)=m
%c vector columna de orden n 
%b vector columna de orden m 
%out 
%x valor columna de orden n con la sol numerica del problema 
%lambda vector columna de orden que representa el mult de lagrange 
%optimización 25 de agosto
%--------------------------------------------------------------
m=length(b); %numero de restricciones 
n=length(c); %numero de variables
K=[Q A'; A zeros(m)];
ld=[-c ; b];
%resolver el sistem lineal k*w=ld
%w=[x;lambda];
w=K\ld;
x=w(1:n);
lambda = w(n+1:n+m);

 end



