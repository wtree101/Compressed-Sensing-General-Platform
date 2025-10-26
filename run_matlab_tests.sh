#!/bin/bash
# Automatic MATLAB Testing Script
# This script runs MATLAB tests automatically from the command line

# Set MATLAB path
MATLAB_PATH="/Applications/MATLAB_R2022b.app/bin/matlab"

# Check if MATLAB exists
if [ ! -f "$MATLAB_PATH" ]; then
    echo "Error: MATLAB not found at $MATLAB_PATH"
    echo "Please update MATLAB_PATH in this script"
    exit 1
fi

# Change to project directory
PROJECT_DIR="/Users/wutong/Documents/MATLAB/GeneralPlatform"
cd "$PROJECT_DIR" || exit 1

echo "=== Automatic MATLAB Testing ==="
echo "Project directory: $PROJECT_DIR"
echo "MATLAB path: $MATLAB_PATH"
echo

# Run diagnostic test
echo "1. Running diagnostic test..."
"$MATLAB_PATH" -nodisplay -nosplash -nodesktop -r "
try
    addpath('utilities');
    addpath('solver'); 
    addpath('Initialization_groundtruth');
    diagnostic_test;
    fprintf('\n✓ Diagnostic test completed successfully\n');
catch ME
    fprintf('\n✗ Diagnostic test failed: %s\n', ME.message);
    exit(1);
end
exit(0);
"

DIAGNOSTIC_RESULT=$?
if [ $DIAGNOSTIC_RESULT -ne 0 ]; then
    echo "Diagnostic test failed. Stopping tests."
    exit 1
fi

# Run simple test
echo
echo "2. Running simple algorithm test..."
"$MATLAB_PATH" -nodisplay -nosplash -nodesktop -r "
try
    addpath('utilities');
    addpath('solver');
    addpath('Initialization_groundtruth');
    simple_test;
    fprintf('\n✓ Simple test completed successfully\n');
catch ME
    fprintf('\n✗ Simple test failed: %s\n', ME.message);
    exit(1);
end
exit(0);
"

SIMPLE_RESULT=$?
if [ $SIMPLE_RESULT -ne 0 ]; then
    echo "Simple test failed."
    exit 1
fi

echo
echo "=== All Tests Passed Successfully! ==="
echo "✓ Diagnostic test: PASSED"
echo "✓ Simple test: PASSED"
echo "✓ Project is ready for use"
