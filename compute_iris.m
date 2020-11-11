function [x1, y1, x2, y2, y3_1, y3_2, counts, missClassNum, missClassRate] = compute_iris(x, y, class, xt, yt, ct, lr, gamma_0)
    %% Function:  LeastSquares Problem Solver => min ||A*omega - t||, A =[1; x; y], t = class
    A = [ones(numel(x), 1) x' y'];
    if gamma_0 ~= 0
        B = (A') * A;
        gamma_updated = balancing_principle(gamma_0, A, class');
        B = B + gamma_updated*(eye(size(B)));
        wls = B\((A')*(class'));
    else
        wls = A\(class');
    end

    y1 = zeros(1,2);
    x1(1)= min(x); x1(2)=max(x);

    for i=1:1:2
        y1(i) =  ((0.5-wls(1))/wls(3)) - ((wls(2)/wls(3))*x1(i));
    end
    
    m = ( y1(1,2) - y1(1,1) ) / (x1(2) - x1(1));
    pred = ct';
    
    mx = mean(x(1,:));
    my = mean(y(1,:));
    b = my/mx;
    
    for i = 1:length(xt)
        if (yt(1,i) - ((m*xt(1,i)) + b) ) > 0     % point is above line
            pred(i) = 1;
        elseif (yt(1,i) - ((m*xt(1,i)) + b) ) < 0 % point is below line
            pred(i) = 0;
        end
    end

    missClassNumLS = (length(pred)-sum(pred == ct'));
    missClassRateLS = missClassNumLS/length(pred);
    fprintf("Number of MissClassified Points : %d\nMissClassification Rate: %.3f\n", missClassNumLS, missClassRateLS);

    %% Function: Perceptron Learning Algorithm: d = 1
    % initialize parameters
    rng(1, 'philox');
    wl = randn(3, 1);
    hyp = zeros(size(x,2), 1)';
    best_wl = wl;
    count = 0;
    pred = ct;
    
    if gamma_0 ~= 0
        gamma_updated = balancing_principle(gamma_0, [ones(numel(x), 1) x' y'], class');
    else
        gamma_updated = 0;
    end

    while sum(class ~= hyp)
        for i = 1:size(x,2)
            if (wl(1) + wl(2)*x(i)  + wl(3)*y(i)) > 0 
                hyp(i) = 1;
            else  
                hyp(i) = 0;
            end
            % save current weights
            best_wl = wl;
            
            % update weights and gamma
            wl(1) = wl(1) + lr*(class(i) - hyp(i));
            wl(2) = wl(2) + lr*( ((class(i) - hyp(i)) * x(i)) + (gamma_updated*best_wl(2)) );
            wl(3) = wl(3) + lr*( ((class(i) - hyp(i)) * y(i)) + (gamma_updated*best_wl(2)) );
        end

        count = count+1;
        if count > 1e10
            break
        end
    end

    y2 = zeros(1,2);
    x2(1)= min(x);    x2(2)=max(x);
    
    for i=1:1:2
        y2(i) = (-best_wl(1)/best_wl(3)) - ((best_wl(2)/best_wl(3))*x2(i));
    end
    
    counts(1) = count-1;
    fprintf('Iterations: %d\n', count-1);
 
    for i=1:length(xt)
        Arow = [1 xt(i) yt(i)];
        if (Arow*best_wl) > 0
            pred(1,i) = 1;
        else
            pred(1,i) = 0;
        end
    end
    
    missClassNumL = (length(pred)-sum(pred == ct));
    missClassRateL = missClassNumL/length(pred);
    
    fprintf("Number of MissClassified Points : %d\nMissClassification Rate: %.3f\n", missClassNumL, missClassRateL);

    %% Function: Perceptron Learning Algorithm: d = 2
    % initialize parameters
    rng(1, 'philox');
    wq = randn(6, 1);
    hyp = zeros(size(x,2), 1)';
    best_wq = wq;
    count = 0;
    pred = ct;
    z1 = pred;
    z2 = z1;
    
    if gamma_0 ~= 0
        gamma_updated = balancing_principle(gamma_0, [ones(numel(x), 1) x' y'], class');
    else
        gamma_updated = 0;
    end

    while sum(class ~= hyp)
        for i=1:size(x,2)
            if wq(1) + wq(2)*x(i)  + wq(3)*y(i)  + ...
                wq(4)*x(i)*x(i)  + wq(5)*x(i)*y(i) + ...
                wq(6)*y(i)*y(i)  > 0 
                hyp(i)=1; 
            else  
                hyp(i)=0;
            end
            % save current weights
            best_wq = wq;
            % update weights
            wq(1) = wq(1) + lr*(class(i) - hyp(i));   
            wq(2) = wq(2) + lr*( ((class(i) - hyp(i)) *x(i)) + (gamma_updated*best_wq(2)) ); 
            wq(3) = wq(3) + lr*( ((class(i) - hyp(i)) *y(i)) + (gamma_updated*best_wq(3)) ); 
            wq(4) = wq(4) + lr*( ((class(i) - hyp(i)) *x(i)*x(i)) + (gamma_updated*best_wq(4)) ); 
            wq(5) = wq(5) + lr*( ((class(i) - hyp(i)) *x(i)*y(i)) + (gamma_updated*best_wq(5)) ); 
            wq(6) = wq(6) + lr*( ((class(i) - hyp(i)) *y(i)*y(i)) + (gamma_updated*best_wq(6)) ); 
        end

        count = count+1;
        if count > 1e10
            break
        end
    end

    q_a = best_wq(6);
    x = sort(x);

    y3_1 = zeros(1, size(x,2));
    y3_2 = zeros(1, size(x,2));

    for i = 1:size(x,2)
        q_b = best_wq(5)*x(i) + best_wq(3);
        q_c = best_wq(1) + best_wq(2)*x(i) + best_wq(4)*x(i)*x(i);
        D  = q_b*q_b - 4*q_a*q_c;
        y3_1(i) = (-q_b + sqrt(D))/(2*q_a);
        y3_2(i) = (-q_b - sqrt(D))/(2*q_a);
    end
    
    counts(2) = count-1;
    fprintf('Iterations: %d\n', count-1);
    
    qa = best_wq(6);
    for i=1:length(xt)
        qb = best_wq(5)*xt(i) + best_wq(3);
        qc = best_wq(1) + best_wq(2)*xt(i) + best_wq(4)*xt(i)*xt(i);
        D  = qb*qb - 4*q_a*qc;
        z1(i) = (-qb + sqrt(D))/(2*qa);
        z2(i) = (-qb - sqrt(D))/(2*qa);
        if (yt(i) > z1(i))
            pred(i) = 1;
        else
            pred(i) = 0;
        end
    end

    missClassNumQ = (length(pred)-sum(pred == ct));
    missClassRateQ = missClassNumQ/length(pred);

    fprintf("Number of MissClassified Points : %d\nMissClassification Rate: %.3f\n", missClassNumQ, missClassRateQ);
    
    missClassNum = [missClassNumLS, missClassNumL, missClassNumQ];
    missClassRate = [missClassRateLS, missClassRateL, missClassRateQ];
end