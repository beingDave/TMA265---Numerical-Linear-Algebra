function [ls_y, pll_x, pll_y, plq_y1, plq_y2] = compute_seal(x, y, class, hyp, gamma_0, lr, CountLimit, Seal_l, Seal_w)
    % Generate Vandemorte Matrix
    fprintf('1. Generating Vandemorte Matrix\n');
    m = size(Seal_w, 1);
    d = 1;
    A = [];
    for i=1:1:m
        for j=1:1:d+1
            A(i,j)=power(x(i),j-1);
            if isnan(Seal_l(i))
                y(i)= 0;
            end
        end
    end

    sch = 0;

    for i=m+1:1:2*m
        sch = sch+1;
        for j=1:1:d+1
            A(i,j)=power(x(i),j-1);
            if isnan(Seal_w(sch))
                y(i)= 0;
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%
    %%%                LEAST SQUARES
    %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('2. Computing Least Squares\n');
    if gamma_0 ~= 0
        B = (A') * A;
        gamma_updated = balancing_principle(gamma_0, A, y');
        B = B + gamma_updated*(eye(size(B)));
        w = B\((A')*(y'));
    else
        w = A\(y');
    end
    
    ls_y = A*w;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%
    %%%                PERCEPTRON LINEAR ALGO
    %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    fprintf('3. Computing Solution by Perceptron "d = 1" Learning Algorithm\n');
    wl = randn(3, 1);
    best_wl = wl;
    count = 0;
    hyp1 = hyp;
    
    if gamma_0 ~= 0
        gamma_updated = balancing_principle(gamma_0, A, y');
    else
        gamma_updated = 0;
    end

    while sum(class ~= hyp1)
        for i = 1:size(x,2)
            if (wl(1) + wl(2)*x(i)  + wl(3)*y(i)) > 0 
                hyp1(i) = 1;
            else  
                hyp1(i) = 0;
            end
            % save current weights
            best_wl = wl;
            
            % update weights and gamma
            wl(1) = wl(1) + lr*(class(i) - hyp1(i));
            wl(2) = wl(2) + lr*( ((class(i) - hyp1(i)) * x(i)) + (gamma_updated*best_wl(2)) );
            wl(3) = wl(3) + lr*( ((class(i) - hyp1(i)) * y(i)) + (gamma_updated*best_wl(2)) );
        end

        count = count+1;
        if count > CountLimit
            break
        end
    end

    pll_y = zeros(1,2);
    pll_x(1)= min(x);    pll_x(2)=max(x);
    
    for i=1:1:2
        pll_y(i) = (-best_wl(1)/best_wl(3)) - ((best_wl(2)/best_wl(3))*pll_x(i));
    end
    
    fprintf('Iterations: %d\n', count-1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%
    %%%                PERCEPTRON QUAD ALGO
    %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('4. Computing Solution by Perceptron "d = 2" Learning Algorithm\n');
    wq = randn(6, 1);
    hyp2 = hyp;
    best_wq = wq;
    count = 0;
    
    if gamma_0 ~= 0
        gamma_updated = balancing_principle(gamma_0, A, y');
    else
        gamma_updated = 0;
    end

    while sum(class ~= hyp2)
        for i=1:size(x,2)
            if wq(1) + wq(2)*x(i)  + wq(3)*y(i)  + ...
                wq(4)*x(i)*x(i)  + wq(5)*x(i)*y(i) + ...
                wq(6)*y(i)*y(i)  > 0 
                hyp2(i)=1; 
            else  
                hyp2(i)=0;
            end
            % save current weights
            best_wq = wq;
            % update weights
            wq(1) = wq(1) + lr*(class(i) - hyp2(i));   
            wq(2) = wq(2) + lr*( ((class(i) - hyp2(i)) *x(i)) + (gamma_updated*best_wq(2)) ); 
            wq(3) = wq(3) + lr*( ((class(i) - hyp2(i)) *y(i)) + (gamma_updated*best_wq(3)) ); 
            wq(4) = wq(4) + lr*( ((class(i) - hyp2(i)) *x(i)*x(i)) + (gamma_updated*best_wq(4)) ); 
            wq(5) = wq(5) + lr*( ((class(i) - hyp2(i)) *x(i)*y(i)) + (gamma_updated*best_wq(5)) ); 
            wq(6) = wq(6) + lr*( ((class(i) - hyp2(i)) *y(i)*y(i)) + (gamma_updated*best_wq(6)) ); 
        end

        count = count+1;
        if count > CountLimit
            break
        end
    end

    q_a = best_wq(6);
    x = sort(x);

    plq_y1 = zeros(1, size(x,2));
    plq_y2 = zeros(1, size(x,2));

    for i = 1:size(x,2)
        q_b = best_wq(5)*x(i) + best_wq(3);
        q_c = best_wq(1) + best_wq(2)*x(i) + best_wq(4)*x(i)*x(i);
        D  = q_b*q_b - 4*q_a*q_c;
        plq_y1(i) = (-q_b + sqrt(D))/(2*q_a);
        plq_y2(i) = (-q_b - sqrt(D))/(2*q_a);
    end
    fprintf('Iterations: %d\n', count-1);
end