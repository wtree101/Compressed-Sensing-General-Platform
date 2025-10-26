function run_all_tests()
    % Run comprehensive test suite for the project
    
    fprintf('=== Running Complete Test Suite ===\n\n');
    
    %% Step 1: Setup paths
    fprintf('1. Setting up project paths...\n');
    setup_project;
    
    %% Step 2: Diagnostic test
    fprintf('\n2. Running diagnostic test...\n');
    try
        diagnostic_test;
        fprintf('✓ Diagnostic test completed\n');
    catch ME
        fprintf('✗ Diagnostic test failed: %s\n', ME.message);
        return;
    end
    
    %% Step 3: Simple test
    fprintf('\n3. Running simple algorithm test...\n');
    try
        simple_test;
        fprintf('✓ Simple test completed\n');
    catch ME
        fprintf('✗ Simple test failed: %s\n', ME.message);
        fprintf('Check the error above for debugging\n');
        return;
    end
    
    %% Summary
    fprintf('\n=== Test Suite Complete ===\n');
    fprintf('✓ All tests passed successfully!\n');
    fprintf('✓ Project is ready for use\n');
    fprintf('\nNext steps:\n');
    fprintf('- Modify parameters in simple_test.m for custom tests\n');
    fprintf('- Run Phasediagram_clean.m for full experiments\n');
    fprintf('- Check TESTING_GUIDE.md for more details\n');
end
