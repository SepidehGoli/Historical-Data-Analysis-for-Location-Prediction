for cluster = 1:number_of_clusters
    cluster
    if(cluster == 4 || cluster == 6)
        continue
    end
    temp = train_clustering(train_clustering(:,27) == cluster, :);
    unique_IDs = unique(temp(:,1));
    numOfVeh = size(unique_IDs, 1);
    train = [];
    for i =1:min(10,numOfVeh)
        train= [train;temp(temp(:,1) == unique_IDs(i),:)];
    end
    
    % Preparing the GP parameters and hyperparameters
    covfunc = @covSEiso;
    likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
    
    % Running GP for vx
    hyp_vx(cluster).cov = [0 ; 0];
    hyp_vx(cluster).lik = log(0.1);
    hyp_vx(cluster) = minimize(hyp_vx(cluster), @gp, -100, @infExact, [], covfunc, likfunc, train(:,5:6), train(:,25));
    exp(hyp_vx(cluster).lik);
    nlml_vx = gp(hyp_vx(cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,25));
    
    % Running GP for vy
    hyp_vy(cluster).cov = [0 ; 0];
    hyp_vy(cluster).lik = log(0.1);
    hyp_vy(cluster) = minimize(hyp_vy(cluster), @gp, -100, @infExact, [], covfunc, likfunc, train(:,5:6), train(:,26));
    exp(hyp_vy(cluster).lik);
    nlml_vy = gp(hyp_vy(cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,26));
    %end
end