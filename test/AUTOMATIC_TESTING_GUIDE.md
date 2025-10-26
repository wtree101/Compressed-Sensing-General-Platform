# Automatic MATLAB Testing Guide

## Available Testing Options

Yes! There are several ways to connect to MATLAB and run automatic testing:

### Option 1: Shell Script (Recommended)
**Best for**: CI/CD, automated workflows, command-line testing

```bash
# Run this from terminal
cd /Users/wutong/Documents/MATLAB/GeneralPlatform
./run_matlab_tests.sh
```

**Features:**
- ✅ No additional dependencies
- ✅ Works with any MATLAB installation
- ✅ Returns proper exit codes for CI/CD
- ✅ Captures and displays output

### Option 2: MATLAB Batch Mode (Simplest)
**Best for**: Quick testing, manual runs

```bash
# Single command testing
/Applications/MATLAB_R2022b.app/bin/matlab -batch "automated_test_suite"

# Or create an alias for convenience
alias matlab-test="cd /Users/wutong/Documents/MATLAB/GeneralPlatform && /Applications/MATLAB_R2022b.app/bin/matlab -batch 'automated_test_suite'"
```

**Features:**
- ✅ Minimal setup required
- ✅ Direct MATLAB execution
- ✅ Suppresses GUI for headless testing

### Option 3: Python-MATLAB Integration
**Best for**: Complex testing workflows, data analysis

```bash
# First install MATLAB engine for Python
pip install matlabengine

# Then run tests
python test_with_python.py
```

**Features:**
- ✅ Full Python integration
- ✅ Advanced error handling
- ✅ Can integrate with pytest, unittest
- ✅ Data analysis and visualization in Python

### Option 4: VS Code Integration
**Best for**: Development and debugging

1. Install MATLAB extension for VS Code
2. Use Command Palette: "MATLAB: Run Current Script"
3. Or use integrated terminal with any of the above methods

## Quick Start

### Method 1: One-Line Test
```bash
cd /Users/wutong/Documents/MATLAB/GeneralPlatform && /Applications/MATLAB_R2022b.app/bin/matlab -batch "addpath('utilities'); addpath('solver'); addpath('Initialization_groundtruth'); simple_test"
```

### Method 2: Using the Shell Script
```bash
cd /Users/wutong/Documents/MATLAB/GeneralPlatform
chmod +x run_matlab_tests.sh
./run_matlab_tests.sh
```

## Expected Output

### Successful Test Run:
```
=== Automatic MATLAB Testing ===
Project directory: /Users/wutong/Documents/MATLAB/GeneralPlatform

1. Running diagnostic test...
✓ All core functions available
✓ onetrial works (final error: 1.23e-04)

2. Running simple algorithm test...
--- Testing GD Algorithm ---
  Final relative error: 0.000123
  Status: CONVERGED ✓
--- Testing RGD Algorithm ---
  Final relative error: 0.000045
  Status: CONVERGED ✓
--- Testing SubGD Algorithm ---
  Final relative error: 0.000234
  Status: CONVERGED ✓

=== All Tests Passed Successfully! ===
✓ Diagnostic test: PASSED
✓ Simple test: PASSED
✓ Project is ready for use
```

## Continuous Integration (CI/CD)

### GitHub Actions Example:
```yaml
name: MATLAB Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup MATLAB
      uses: matlab-actions/setup-matlab@v1
    - name: Run tests
      run: |
        cd GeneralPlatform
        matlab -batch "automated_test_suite"
```

## Troubleshooting

### Common Issues:

1. **"matlab command not found"**
   ```bash
   # Add MATLAB to PATH
   export PATH="/Applications/MATLAB_R2022b.app/bin:$PATH"
   ```

2. **"Function not found" errors**
   ```bash
   # Ensure you're in the project directory
   cd /Users/wutong/Documents/MATLAB/GeneralPlatform
   ```

3. **Permission denied**
   ```bash
   chmod +x run_matlab_tests.sh
   ```

4. **MATLAB engine installation for Python**
   ```bash
   # Navigate to MATLAB installation
   cd /Applications/MATLAB_R2022b.app/extern/engines/python
   python setup.py install
   ```

## Performance Benchmarks

- **Diagnostic test**: ~10 seconds
- **Simple test**: ~30 seconds  
- **Full test suite**: ~45 seconds
- **Shell script overhead**: ~5 seconds

## Integration with Development Workflow

### Pre-commit Hook:
```bash
#!/bin/sh
# .git/hooks/pre-commit
cd /Users/wutong/Documents/MATLAB/GeneralPlatform
./run_matlab_tests.sh
exit $?
```

### Makefile Integration:
```makefile
test:
	cd /Users/wutong/Documents/MATLAB/GeneralPlatform && ./run_matlab_tests.sh

quick-test:
	cd /Users/wutong/Documents/MATLAB/GeneralPlatform && /Applications/MATLAB_R2022b.app/bin/matlab -batch "simple_test"
```
