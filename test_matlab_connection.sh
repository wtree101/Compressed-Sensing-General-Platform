#!/bin/bash
# Simple MATLAB connectivity test

MATLAB_PATH="/Applications/MATLAB_R2022b.app/bin/matlab"
PROJECT_DIR="/Users/wutong/Documents/MATLAB/GeneralPlatform"

echo "=== Testing MATLAB Connectivity ==="
echo "MATLAB path: $MATLAB_PATH"
echo "Project directory: $PROJECT_DIR"

# Test 1: Check if MATLAB executable exists
if [ ! -f "$MATLAB_PATH" ]; then
    echo "❌ MATLAB not found at $MATLAB_PATH"
    exit 1
else
    echo "✅ MATLAB executable found"
fi

# Test 2: Test basic MATLAB command
echo "Testing basic MATLAB execution..."
cd "$PROJECT_DIR"

"$MATLAB_PATH" -nodisplay -nosplash -nodesktop -r "
fprintf('✅ MATLAB is working!\n');
fprintf('Current directory: %s\n', pwd);
fprintf('MATLAB version: %s\n', version);
exit(0);
"

if [ $? -eq 0 ]; then
    echo "✅ MATLAB connectivity test passed!"
    echo ""
    echo "You can now run automatic tests with:"
    echo "  ./run_matlab_tests.sh"
    echo "  OR"
    echo "  $MATLAB_PATH -batch \"automated_test_suite\""
else
    echo "❌ MATLAB connectivity test failed"
    exit 1
fi
