function save_experiment_point(save_dir, r, m, mu, trial_num, success_rate, output, T, init_method, add_flag)
    % SAVE_TENSOR_EXPERIMENT_POINT Save single tensor experiment point with merge capability
    %
    % Inputs:
    %   save_dir     - Directory to save results
    %   r            - Rank value
    %   m            - Number of measurements
    %   mu           - Step size
    %   trial_num    - Number of trials
    %   success_rate - Success probability (0 to 1)
    %   output       - Error history array (T×1)
    %   T            - Number of iterations
    %   init_method  - Initialization method used
    %   add_flag     - 0: overwrite, 1: merge with existing
    %
    % This function saves individual point data in the format:
    %   r_<r>_m_<m>_t_<trial_num>.mat containing 'point' structure
    %
    % Point structure fields:
    %   .r           - Rank
    %   .m           - Number of measurements
    %   .mu          - Step size
    %   .trial_num   - Total number of trials
    %   .p           - Success probability
    %   .e           - Error history (averaged across trials)
    %   .T           - Number of iterations
    %   .init_method - Initialization method
    
    %% Create point structure
    point = struct();
    point.r = r;
    point.m = m;
    point.mu = mu;
    point.trial_num = trial_num;
    point.p = success_rate;  % Success probability
    point.e = output;  % Error history
    point.T = T;
    point.init_method = init_method;
    
    %% Determine point filename
    point_name = sprintf('r_%d_m_%d_t_%d.mat', r, m, trial_num);
    point_file = fullfile(save_dir, point_name);
    
    %% Save or merge based on add_flag
    if add_flag == 0
        % Overwrite mode: simply save new data
        save(point_file, 'point');
        % fprintf('    Saved: %s\n', point_name);
    else
        % Merge mode: load existing and combine
        if exist(point_file, 'file')
            try
                old_data = load(point_file, 'point');
                data_old = old_data.point;
                
                % Calculate total trials
                total_trials = data_old.trial_num + point.trial_num;
                
                % Update probability: weighted average
                p_new = (data_old.p * data_old.trial_num + point.p * point.trial_num) / total_trials;
                
                % Update error history: weighted average
                e_new = (data_old.e * data_old.trial_num + point.e * point.trial_num) / total_trials;
                
                % Update the point structure
                point.p = p_new;
                point.e = e_new;
                point.trial_num = total_trials;
                
                fprintf('    Merged: %s (now %d trials, success: %.1f%%)\n', ...
                        point_name, total_trials, p_new*100);
            catch ME
                fprintf('    Warning: Failed to merge %s: %s. Saving as new.\n', ...
                        point_name, ME.message);
            end
        else
            % fprintf('    Saved new: %s\n', point_name);
        end
        
        save(point_file, 'point');
    end
end
