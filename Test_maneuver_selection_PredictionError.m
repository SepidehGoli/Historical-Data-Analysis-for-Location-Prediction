% Testing without maneuver detection

num_of_test_trajs = 10;
num_of_train_trajs = 10;
start_time = 10;
end_time = 40;

% Euclidean Distances
% 3-D matrices
error_gp = zeros(number_of_clusters,num_of_test_trajs, end_time);
error_dyn = zeros(number_of_clusters,num_of_test_trajs, end_time);

for cluster = 1:number_of_clusters
    cluster
    if(cluster == 4 || cluster == 6 )
        continue
    end
    
    temp = train_clustering(train_clustering(:,27) == cluster, :);
    unique_IDs = unique(temp(:,1));
    numOfVeh = size(unique_IDs, 1);
    
    train = [];
    for i =1:min(num_of_train_trajs,numOfVeh)
        train= [train;temp(temp(:,1) == unique_IDs(i),:)];
    end
    
    test = [];
    figure
    start_traj = num_of_train_trajs + 1;
    end_traj = num_of_test_trajs + num_of_train_trajs;
    for i = start_traj : min(end_traj,numOfVeh) % trajectories from each cluster for testing
        test= temp(temp(:,1) == unique_IDs(i),:);
        scatter(test(:,5),test(:,6), 'b.');
        hold on;
        
        steps = size(test,1);
        steps = min(end_time,steps);
        if size(test,1) < (start_time + 1)
            continue
            % go to the next traj
        end
        
        % maneuver selection
        p_model = zeros(1,number_of_clusters);
        for c = 1:number_of_clusters
            if c == 4 || c == 6
                p_model(c) = Inf;
                continue;
            end
            
            % preparing train data of cluster c
            temp_c = train_clustering(train_clustering(:,27) == c, :);
            unique_IDs_c = unique(temp_c(:,1));
            numOfVeh_c = size(unique_IDs_c, 1);
            train = [];
            for j =1:min(num_of_train_trajs,numOfVeh_c)
                train= [train;temp_c(temp_c(:,1) == unique_IDs_c(j),:)];
            end
            
            [m_vx,s_vx,m_vx1,s_vx1,lx]= gp(hyp_vx(c), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,25),test(1:start_time,5:6));
            [m_vy,s_vy,m_vy1,s_vy1,ly] = gp(hyp_vy(c), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,26),test(1:start_time,5:6));
            
            % Initialization of position, velocity and acceleration
            p = [test(1,5),test(1,6)];
            for j = 2:start_time
                p = Project(p,[m_vx(j-1),m_vy(j-1)]);
                %calculating Euclidean distances
                error = EuclideanD(p, test(j,5:6));
                p_model(c) = p_model(c) + error;
            end
        end
        [val,index] = min(p_model);
        selected_cluster = index
        
        temp_c = train_clustering(train_clustering(:,27) == selected_cluster, :);
        unique_IDs_c = unique(temp_c(:,1));
        numOfVeh_c = size(unique_IDs_c, 1);
        train = [];
        for j =1:min(num_of_train_trajs,numOfVeh_c)
            train= [train;temp_c(temp_c(:,1) == unique_IDs_c(j),:)];
        end
        
        [m_vx s_vx]= gp(hyp_vx(selected_cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,25),test(1:steps,5:6));
        [m_vy s_vy] = gp(hyp_vy(selected_cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,26),test(1:steps,5:6));
        
        % Initialization of position, velocity and acceleration
        p = [test(start_time,5),test(start_time,6)];
        v = [test(start_time,25),test(start_time,26)];
        a = [test(start_time,25) - test(start_time - 1,25) , test(start_time,26) - test(start_time - 1,26)];
        
        p_gp = p;
        p_dyn = p;
        v_dyn = v;
        
        for j = start_time:steps
            [m_vx_val s_vx]= gp(hyp_vx(selected_cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,25),p);
            [m_vy_val s_vy] = gp(hyp_vy(selected_cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,26),p);
            p_gp = Project(p_gp,[m_vx_val,m_vy_val]);
            [p_dyn,v_dyn] =  Project(p_dyn,v_dyn,a);
            
            %calculating Euclidean distances
            error_gp(selected_cluster,i-num_of_train_trajs,j) = EuclideanD(p_gp, test(j,5:6));
            error_dyn(selected_cluster,i-num_of_train_trajs,j) = EuclideanD(p_dyn, test(j,5:6));
        end
        
    end
    hold off;
end


figure
error_gp(error_gp == 0) = NaN;
error_dyn(error_dyn == 0) = NaN;

n = number_of_clusters * num_of_test_trajs;

sd = std(reshape(error_gp, n, end_time), 'omitnan');
errorbar(mean(reshape(error_gp, n, end_time), 'omitnan'),sd,'-s','MarkerSize',5);
hold on;
sd = std(reshape(error_dyn, n, end_time), 'omitnan');
errorbar(mean(reshape(error_dyn, n, end_time), 'omitnan'),sd,'-s','MarkerSize',5);
hold off;

% legend('gp1', 'gp2','dynamic','ave gp1 gp2', 'ave gp1 gp2 dyn', 'ave gp2 dyn');
legend('gp', 'dynamic');