function gamma_updated = balancing_principle(gamma, A, t)
    %% Implementation
    gamma_updated = 0;
    C = 1;                          % zero-crossing method
    theta = 0.01;                   % tolerance 
    count = 1;
    check_condition = 1;
    while (check_condition ~= 0)
        % Step 2: Compute Value Function 'J' to obtain omega 'w'
        % J = 0.5 * ((norm( (Aw - t), 2)^2) +  gamma*(norm(w, 2))^2);
        % 0.5 [ (2A')*(Aw - t) + (2*gamma*w) ] = 0
        % (A')Aw - (A')t + gamma*w = 0
        % ((A')A + gamma*I) w = (A')t
        % w = inv((A')A + gamma*I) * (A')t
        w = ( (A')*A + gamma*eye(size((A')*A)) ) \ ((A')*t);
        phi_bar = (A')*(A*w - t);
        psi_bar = 2*w;

        % Step 3: Update Regularization Parameter
        gamma_updated = C * (norm( phi_bar )^2 / norm( psi_bar )^2);
        count = count+1;
        if (abs(gamma_updated - gamma) > theta)
            gamma = gamma_updated;
        else
            check_condition = 0;
        end
        if count > 100
            break;
        end
    end
    fprintf('Regularization Iteration: %d\ngamma: %.12f\n', count, gamma_updated);
end