function turning_point = PhaseTransition(d1,d2,r,kappa,trial_num,verbose)
    if nargin < 6
        verbose = 0; % Set default value for 'verbose' if not provided
    end
    Xstar = groundtruth(d1,d2,r,kappa);
    
    
    % m = C * gap * r(d1 + d2)  C 3-4?
    freedom = r*(d1 + d2);
    
    
    C_max = 20; 
    gap = 0.5;
    
    success_graph = zeros(1,C_max);
    parfor C = 1:C_max
        success_rate = multipletrial(floor(C*freedom*gap),d1,d2,r,1,trial_num,0,Xstar);
        success_graph(C) = success_rate;
    end
    
    cmin = 0; cmax= 0;
    for i = 1:C_max
        if success_graph(i)==0
            cmin = i;
        end
        if success_graph(C_max - i + 1)==1
            cmax = C_max - i + 1;
        end
    end
    %turning_point = (cmin + cmax)/2;
    turning_point = floor(cmax*freedom*gap);
    
    
    if verbose==1
        fig = figure('Visible', 'off');
        plot((1:C_max)*gap, success_graph, '-o', 'LineWidth', 2, 'MarkerSize', 8); % Line with markers
        grid on; % Add grid for better visualization
        xlabel('$m/r*(d_1+d_2)$', 'FontSize', 12); % X-axis label
        ylabel('Success Rate (%)', 'FontSize', 12); % Y-axis label
    
        % Save the figure
        output_file = sprintf('success_rate_d1_%d_d2_%d_r_%d_kappa_%d_trial_%d', d1, d2, r, kappa,trial_num);
        saveas(fig, ['figures/',output_file,'.png']); % Save the figure as a PNG file
        % saveas(fig, ['figures_fig/',output_file,'.fig']);
        % disp(['Figure saved as ', output_file]);
         %save the data for graph
        data_file = sprintf('success_rate_data_d1_%d_d2_%d_r_%d_kappa_%d_trial_%d.mat', d1, d2, r, kappa,trial_num);
        m_divide_freedom = (1:C_max) * gap; % X-axis data
        
        save(['data/', data_file], 'm_divide_freedom', 'success_graph');
        disp(['Graph data saved as ', ['data/', data_file]]);
    end
    
    
    
    end
    
    