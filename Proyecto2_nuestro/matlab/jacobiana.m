function [jh] = jacobiana(hx, x0)
% Calcula la matriz jacobiana de hx: R^n --> R^m

    n = length(x0);
    h = feval(hx, x0);
    m = length(h);
    
    jh = zeros(m, n);
    ep = 1e-5;
    
    for j = 1:n
        y = x0;
        y(j) = y(j) + ep;
        hy = feval(hx, y);
        jh(:, j) = (hy - h) / ep;
    end

end