num_of_test_trajs = 10;
num_of_train_trajs = 10;

error_gp1 = zeros(number_of_clusters,num_of_test_trajs, 40);
error_gp2 = zeros(number_of_clusters,num_of_test_trajs, 40);
error_dyn = zeros(number_of_clusters,num_of_test_trajs, 40);
erro_ave_gp = zeros(number_of_clusters,num_of_test_trajs,40);
error_ave_gp_dyn = zeros(number_of_clusters,num_of_test_trajs,40);
error_ave_gp2_dyn = zeros(number_of_clusters,num_of_test_trajs,40);

for cluster = 1:number_of_clusters
    if cluster == 2 || cluster == 6 || cluster == 7
        continue
    end
    cluster
    temp = train_clustering(train_clustering(:,27) == cluster, :);
    unique_IDs = unique(temp(:,1));
    numOfVeh = size(unique_IDs, 1);
    
    train = [];
    for i =1:min(10,numOfVeh)
        train= [train;temp(temp(:,1) == unique_IDs(i),:)];
%         scatter(train(:,5),train(:,6), 'b.');
%         hold on;
    end
    
    test = [];
    
    figure
    start_traj = num_of_train_trajs + 1;
    end_traj = num_of_test_trajs + num_of_train_trajs;
    for i = start_traj : min(end_traj,numOfVeh) % 5 trajectories from each cluster for testing
        test= temp(temp(:,1) == unique_IDs(i),:);
        scatter(test(:,5),test(:,6), 'b.');
        hold on;
    
        steps = size(test,1);
        steps = min(40,steps);
        [m_vx s_vx]= gp(hyp_vx(cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,25),test(1:steps,5:6));
        [m_vy s_vy] = gp(hyp_vy(cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,26),test(1:steps,5:6));
        if size(test,1) < 11
            continue
        end
        % Projection
        p = [test(10,5),test(10,6)];
        v = [test(10,25),test(10,26)];
        a = [test(10,25) - test(9,25) , test(10,26) - test(9,26)];
        
        p_gp_1 = p;
        p_gp_2 = p;
        p_dyn = p;
        v_dyn = v;
        
        p_ave_gp = p;
        p_ave_gp_dyn = p;
        p_ave_gp2_dyn = p;
        
        for j = 10:steps
            [m_vx_val s_vx]= gp(hyp_vx(cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,25),p);
            [m_vy_val s_vy] = gp(hyp_vy(cluster), @infExact, [], covfunc, likfunc, train(:,5:6), train(:,26),p);
            p_gp_1 = Project(p_gp_1,[m_vx_val,m_vy_val]);
            p_gp_2 = Project(p_gp_2,[m_vx(j),m_vy(j)]);
            [p_dyn,v_dyn] =  Project(p_dyn,v_dyn,a);
            p_ave_gp = mean([p_gp_1;p_gp_2]);
            p_ave_gp_dyn = mean([p_gp_1;p_gp_2;p_dyn]);
            p_ave_gp2_dyn = mean([p_gp_2;p_dyn]);
            error_gp1(cluster,i-10,j) = EuclideanD(p_gp_1, test(j,5:6));
            error_gp2(cluster,i-10,j) = EuclideanD(p_gp_2, test(j,5:6));
            error_dyn(cluster,i-10,j) = EuclideanD(p_dyn, test(j,5:6));
            erro_ave_gp(cluster,i-10,j) = EuclideanD(p_ave_gp, test(j,5:6));
            error_ave_gp_dyn(cluster,i-10,j) = EuclideanD(p_ave_gp_dyn, test(j,5:6));
            error_ave_gp2_dyn(cluster,i-10,j) = EuclideanD(p_ave_gp2_dyn, test(j,5:6));
%             scatter(p(1),p(2),'g.');
%             hold on;
%             scatter(p2(1),p2(2),'ko');
%             hold on;
%             scatter(p1(1),p1(2),'m.');
%             hold on;
        end
        
    end
    hold off;
end

figure 
error_gp1(error_gp1 == 0) = NaN;
error_gp2(error_gp2 == 0) = NaN;
error_dyn(error_dyn == 0) = NaN;
erro_ave_gp(erro_ave_gp == 0) = NaN;
error_ave_gp_dyn(error_ave_gp_dyn == 0) = NaN;
error_ave_gp2_dyn(error_ave_gp2_dyn == 0) = NaN;

plot(reshape(mean(mean(error_gp1, 'omitnan')),1,40));
hold on;
plot(reshape(mean(mean(error_gp2, 'omitnan')),1,40));
hold on;
plot(reshape(mean(mean(error_dyn, 'omitnan')),1,40));
hold on;
plot(reshape(mean(mean(erro_ave_gp, 'omitnan')),1,40));
hold on;
plot(reshape(mean(mean(error_ave_gp_dyn, 'omitnan')),1,40));
hold on;
plot(reshape(mean(mean(error_ave_gp2_dyn, 'omitnan')),1,40));
hold on;
legend('gp1', 'gp2','dynamic','ave gp1 gp2', 'ave gp1 gp2 dyn', 'ave gp2 dyn');