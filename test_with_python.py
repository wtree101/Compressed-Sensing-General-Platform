#!/usr/bin/env python3
"""
Automatic MATLAB Testing with Python
Requires: matlab.engine (pip install matlabengine)
"""

import matlab.engine
import os
import sys
from pathlib import Path

def setup_matlab_paths(eng, project_dir):
    """Add necessary paths to MATLAB"""
    try:
        eng.addpath(str(project_dir), nargout=0)
        eng.addpath(str(project_dir / 'utilities'), nargout=0)
        eng.addpath(str(project_dir / 'solver'), nargout=0)
        eng.addpath(str(project_dir / 'Initialization_groundtruth'), nargout=0)
        print("✓ MATLAB paths configured")
        return True
    except Exception as e:
        print(f"✗ Failed to set MATLAB paths: {e}")
        return False

def run_diagnostic_test(eng):
    """Run diagnostic test"""
    try:
        print("Running diagnostic test...")
        eng.diagnostic_test(nargout=0)
        print("✓ Diagnostic test passed")
        return True
    except Exception as e:
        print(f"✗ Diagnostic test failed: {e}")
        return False

def run_simple_test(eng):
    """Run simple algorithm test"""
    try:
        print("Running simple algorithm test...")
        eng.simple_test(nargout=0)
        print("✓ Simple test passed")
        return True
    except Exception as e:
        print(f"✗ Simple test failed: {e}")
        return False

def main():
    print("=== Automatic MATLAB Testing with Python ===")
    
    # Project directory
    project_dir = Path("/Users/wutong/Documents/MATLAB/GeneralPlatform")
    if not project_dir.exists():
        print(f"✗ Project directory not found: {project_dir}")
        sys.exit(1)
    
    # Change to project directory
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")
    
    try:
        # Start MATLAB engine
        print("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        print("✓ MATLAB engine started")
        
        # Setup paths
        if not setup_matlab_paths(eng, project_dir):
            eng.quit()
            sys.exit(1)
        
        # Run tests
        tests_passed = 0
        total_tests = 2
        
        if run_diagnostic_test(eng):
            tests_passed += 1
        
        if run_simple_test(eng):
            tests_passed += 1
        
        # Cleanup
        eng.quit()
        print("✓ MATLAB engine stopped")
        
        # Summary
        print(f"\n=== Test Results ===")
        print(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print("❌ Some tests failed")
            sys.exit(1)
            
    except matlab.engine.EngineError as e:
        print(f"✗ MATLAB engine error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
