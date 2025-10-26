%%%%%%%%%% Setup Project Paths
% Add all necessary directories to MATLAB path

% Get the current directory (should be GeneralPlatform)
current_dir = pwd;

% Add subdirectories to path
addpath(current_dir);
addpath(fullfile(current_dir, 'utilities'));
addpath(fullfile(current_dir, 'solver'));
addpath(fullfile(current_dir, 'Initialization_groundtruth'));

% Verify paths are added
fprintf('MATLAB paths added:\n');
fprintf('  %s\n', current_dir);
fprintf('  %s\n', fullfile(current_dir, 'utilities'));
fprintf('  %s\n', fullfile(current_dir, 'solver'));
fprintf('  %s\n', fullfile(current_dir, 'Initialization_groundtruth'));

fprintf('\nProject setup complete! Ready to run tests.\n');
