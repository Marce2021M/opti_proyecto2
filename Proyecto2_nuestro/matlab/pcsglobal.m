function[x,L,k]=pcsglobal(fx,hx,x0)
% Método de programación Cuadrática Sucesiva con búsqueda lineal,
% usando la función de mérito L-1 y actualización de la hessiana
% con la fórmula BF GS para el problema
% Min fx
% Sujeto a hx = 0
%
% fx y hx son cadenas de caracteres con las funciones en Matlab
% de la función objetivo y las restricciones del problema
% El vector x0 es el valor inicial
% Salida
% x.- aproximación al mínimo local
% L.- multiplicador de Lagrange asociado a x.
% k.- número de iteraciones realizadas.
%
% En este código se utilizo las funciones gradiente, jacobiana y pc para
% resolver algunas partes del codigo.
n=length(x0);
m=length(hx(x0));
tol=1e-5; %tolerancia
maxk=100; %máximo de iteraciónes
c1=1e-2;
C0=1;
x=x0;
B=eye(n); %definir B al principio
L=zeros(m,1); %definir el lagrangeano al principio
k=0;
vk=[LAk(fx,hx,x,L);hx(x)];
Cmax=10e5;
Ck=C0;
while (k<=maxk && norm(vk,1)>=tol) %condición de paro del método.
    [pk, L]=pc(B,jacobiana(hx,x),gradiente(fx,x),-hx(x)) %resolver el subproblema cuadrático utilizando programación cuadrática
    if Dk(fx,hx,x,Ck,pk)<0 %elección de Ck
        Ck=Ck;
    else
        Ck=min(Cmax,abs(gradiente(fx,x)'*pk)/norm(hx(x),1)+1);
    end
    alpha=1;
    while (phi(fx,hx,x+alpha*pk,Ck)>phi(fx,hx,x,Ck)+alpha*c1*Dk(fx,hx,x,Ck,pk)) %Usar line search para calcular alpha 
        alpha=alpha/2;
    end
    aux=x; %actualizaciones para actualizar B
    x=x+alpha*pk;
    s=x-aux;
    y=LAk(fx,hx,x,L)-LAk(fx,hx,aux,L);
    if (s'*y<=(0.2)*(s'*B*s)) %actualización de B utilizando BFGS
        theta=(0.8*(s'*B*s))/(s'*B*s-s'*y);
        r=theta*y+(1-theta)*B*s;
    else
        r=y;
    end
    B=B-(B*s*s'*B)/(s'*B*s)+(r'*r)/(s'*r);
    if cond(B)>10e4
       B=eye(n); 
    end
    L=-inv(jacobiana(hx,x)*jacobiana(hx,x)')*jacobiana(hx,x)*gradiente(fx,x); %solución del problema en el inciso f)
    k=k+1;
    vk=[LAk(fx,hx,x,L);hx(x)];
end
end

function[z]=phi(fx,hx,xk,ck) %función que calcula la función de mérito
    z=fx(xk)+ck*norm(hx(xk),1);
end

function[z]=Dk(fx,hx,xk,ck,pk) %función que calcula la función Dpk*phi definida en clase
    z=gradiente(fx,xk).*pk-ck*norm(hx(xk),1);
end

function[z]=LAk(fx,hx,xk,Lk) %función que calcula el gradiente del lagrangeano.
    z=gradiente(fx,xk)+jacobiana(hx,xk)'*Lk;
end
    
    
    
