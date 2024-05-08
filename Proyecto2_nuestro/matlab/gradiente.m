function [gf] = gradiente(fx, x0)
% Calcula por diferencias hacia adelante el gradiente de 
% fx: R^n --> R
%
% (gf)_k = parcial de fx / parcial x_k

    n = length(x0);
    gf = zeros(n, 1);
    ep = 1e-5;
    f0 = feval(fx, x0);
    
    for k = 1:n
        x0(k) = x0(k) + ep;
        gf(k) = (feval(fx,x0) - f0) / ep;
        x0(k) = x0(k) - ep;
    end

end