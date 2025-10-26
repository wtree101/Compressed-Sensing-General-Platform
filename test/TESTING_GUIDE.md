# Testing Guide for Low-Rank Matrix Recovery Project

## Quick Start Testing

### Step 1: Setup Project Paths
```matlab
cd /path/to/GeneralPlatform
setup_project
```

### Step 2: Run Diagnostic Test
```matlab
diagnostic_test
```
This will check if all required functions are available and working.

### Step 3: Run Simple Test
```matlab
simple_test
```
This runs a quick test with small parameters to verify everything works.

### Step 4: Run Full Phase Diagram (Optional)
```matlab
Phasediagram_clean
```
This runs the full experiment (takes longer).

## Test Components

### 1. `diagnostic_test.m`
- **Purpose**: Check if all dependencies are available
- **Runtime**: ~10 seconds
- **Output**: Status of all required functions

### 2. `simple_test.m` 
- **Purpose**: Quick algorithm comparison
- **Parameters**: 20x20 matrices, 50 iterations, 3 algorithms
- **Runtime**: ~30 seconds
- **Output**: Convergence plots and comparison table

### 3. `Phasediagram_clean.m`
- **Purpose**: Full phase transition experiments
- **Parameters**: Configurable in script
- **Runtime**: Minutes to hours depending on settings
- **Output**: Saved results for phase diagram plotting

## Troubleshooting

### Common Issues:

1. **"Function not found" errors**
   - Run `setup_project` first
   - Check that you're in the GeneralPlatform directory

2. **"Solver failed" errors**
   - Check parameter values (mu, lambda, etc.)
   - Try reducing problem size in simple_test.m

3. **"No convergence" warnings**
   - Normal for some parameter combinations
   - Check if final error is reasonable (<1e-2 is good)

### What Good Output Looks Like:

```
=== Algorithm Comparison ===
Algorithm   Final Error   Time (s)   Status
------------------------------------------
GD          1.23e-04     0.045      Pass
RGD         2.15e-05     0.032      Pass  
SubGD       5.67e-04     0.041      Pass
```

### Expected File Structure:
```
GeneralPlatform/
├── utilities/           # Support functions
├── solver/             # Algorithm implementations  
├── Initialization_groundtruth/  # Initialization functions
├── simple_test.m       # Quick test script
├── diagnostic_test.m   # Dependency checker
├── setup_project.m     # Path setup
└── Phasediagram_clean.m # Main experiment
```

## Customizing Tests

### Modify Test Parameters in `simple_test.m`:
```matlab
d1 = 20; d2 = 20;  % Matrix size (increase for harder problems)
m = 100;           % Measurements (increase for better recovery)
T = 50;            # Iterations (increase for better convergence)
r = 2;             # Rank (increase for harder problems)
```

### Test Different Algorithms:
```matlab
algorithms = [0, 1, 2, 3]; % GD, RGD, SGD, SubGD
```

### Test Different Problems:
```matlab
problem_flag = 1; % Change to test phase retrieval
```

## Performance Benchmarks

### Expected Performance (simple_test defaults):
- **Matrix size**: 20×20
- **Measurements**: 100
- **Rank**: 2
- **Expected final error**: < 1e-3
- **Runtime per algorithm**: < 0.1 seconds

### Scaling Guidelines:
- **Runtime scales**: O(T × m × d₁ × d₂)
- **Memory scales**: O(m × d₁ × d₂)
- **For d₁=d₂=60, m=1000**: Expect ~1-10 seconds per trial
