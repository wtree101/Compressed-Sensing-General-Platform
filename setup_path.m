% Setup MATLAB path for Compressed-Sensing-General-Platform
% Add all subdirectories to the MATLAB path

% Get the directory where this script is located
script_dir = fileparts(mfilename('fullpath'));

% Add all subdirectories to the path
addpath(genpath(script_dir));

% Save the path for future MATLAB sessions
savepath;

fprintf('Successfully added all folders in %s to MATLAB path.\n', script_dir);
fprintf('Path has been saved for future sessions.\n');
