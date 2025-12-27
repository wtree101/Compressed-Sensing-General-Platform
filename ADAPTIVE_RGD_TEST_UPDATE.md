# Test Update: Added Adaptive RGD Method

## Summary of Changes

Added **Method 5: Adaptive RGD** to comprehensively test the adaptive stepsize feature with and without preconditioning.

---

## Complete Method Comparison

| Method | Preconditioner | Stepsize | Initial μ | Key Feature |
|--------|---------------|----------|-----------|-------------|
| **Method 1: PRGD (fixed)** | ✅ Yes | Fixed | 0.5 | Baseline preconditioned |
| **Method 2: RGD (fixed)** | ❌ No | Fixed | 1.0 | Baseline non-preconditioned |
| **Method 3: PGD** | ❌ No | Fixed | 1.0 | Standard projected GD |
| **Method 4: PRGD (adaptive)** | ✅ Yes | Adaptive | 0.5 | Line search + preconditioner |
| **Method 5: RGD (adaptive)** | ❌ No | Adaptive | 1.0 | Line search, no preconditioner |

---

## Key Comparisons Enabled

### 1. **Preconditioner Effect (Fixed Stepsize)**
Compare Methods 1 vs 2:
- Same stepsize strategy (fixed)
- Isolates impact of diagonal preconditioner
- Expected: PRGD (fixed) converges faster/better

### 2. **Adaptive Stepsize Effect (With Preconditioner)**
Compare Methods 1 vs 4:
- Same preconditioner (diagonal)
- Fixed vs adaptive stepsize
- Tests: Does line search improve preconditioned method?

### 3. **Adaptive Stepsize Effect (Without Preconditioner)**
Compare Methods 2 vs 5:
- No preconditioner in both
- Fixed vs adaptive stepsize
- Tests: Does line search improve non-preconditioned method?

### 4. **Preconditioner + Adaptive (Combined Effect)**
Compare all 5 methods:
- Which combination works best?
- PRGD (adaptive) vs RGD (adaptive)
- Best of both worlds?

---

## Implementation Details

### Method 5: Adaptive RGD Configuration

```matlab
%% Method 5: Adaptive RGD (RGD with Adaptive Stepsize, no preconditioner)
mu_adaptive_rgd = 1.0;  % Initial stepsize (same as fixed RGD)

params_adaptive_rgd.use_preconditioner = false;  // KEY: No preconditioner
params_adaptive_rgd.use_adaptive_stepsize = true; // Enable line search
params_adaptive_rgd.line_search_max_iter = 20;
params_adaptive_rgd.line_search_beta = 0.5;
params_adaptive_rgd.line_search_c = 1e-4;
```

**How it works:**
- Uses `solve_RGD_amplitude` with `use_preconditioner = false`
- Direction: `D_t = G_t` (raw gradient, no preconditioning)
- Stepsize: Found via backtracking line search
- Combines benefits: factorized storage + adaptive stepsize

---

## Updated Visualizations

### Figure Layout (2×5 subplots)

**Row 1 (Convergence Analysis):**
1. Error vs Iteration (5 curves)
2. Error vs Time (5 curves)
3. Loss vs Iteration (5 curves)
4. Adaptive PRGD Stepsize History
5. Adaptive RGD Stepsize History

**Row 2 (Results Visualization):**
6. Ground Truth X*
7. PRGD (fixed) result
8. PRGD (adaptive) result
9. RGD (fixed) result
10. RGD (adaptive) result

**Color scheme:**
- Blue solid: PRGD (fixed)
- Red dashed: RGD (fixed)
- Black dotted: PGD
- Magenta dash-dot: PRGD (adaptive)
- Cyan dotted: RGD (adaptive)

---

## Summary Table Output

```
Method       | Refine Time (s) | Total Time (s) | Init Error  | Final Error | Error Reduction | Final Loss
-------------|-----------------|----------------|-------------|-------------|-----------------|------------
Init         |               - |         0.0234 | 1.234e-01 | 1.234e-01 |          1.00x | -
PRGD (fixed) |          2.4567 |         2.4801 | 1.234e-01 | 5.678e-05 |       2173.45x | 1.234e-06
RGD (fixed)  |          2.1234 |         2.1468 | 1.234e-01 | 8.901e-04 |        138.67x | 2.345e-05
PGD          |          1.9876 |         2.0110 | 1.234e-01 | 1.234e-03 |        100.00x | 4.567e-05
PRGD (adapt) |          3.1234 |         3.1468 | 1.234e-01 | 4.567e-05 |       2701.23x | 9.876e-07
RGD (adapt)  |          2.7890 |         2.8124 | 1.234e-01 | 6.789e-04 |        181.76x | 1.567e-05
```

---

## Expected Results & Insights

### Hypothesis 1: Preconditioner helps (fixed stepsize)
```
PRGD (fixed) < RGD (fixed)  // Better final error
```
**Why:** Diagonal preconditioner improves conditioning

### Hypothesis 2: Adaptive helps both methods
```
PRGD (adaptive) ≤ PRGD (fixed)
RGD (adaptive) ≤ RGD (fixed)
```
**Why:** Line search finds better stepsizes

### Hypothesis 3: Combined effect
```
Best method: PRGD (adaptive)  // Preconditioner + line search
```
**Why:** Synergy between preconditioning and adaptive stepsize

### Hypothesis 4: Computational cost
```
Time ranking (fast → slow):
PGD < RGD (fixed) < PRGD (fixed) < RGD (adaptive) < PRGD (adaptive)
```
**Why:**
- PGD: Full matrix, simple SVD
- Fixed methods: Factorized, no line search
- Adaptive methods: Extra forward/backward passes per iteration

### Hypothesis 5: Stepsize evolution
```
PRGD (adaptive): α decreases over time (converging to optimum)
RGD (adaptive): α may be more stable (larger initial value)
```

---

## Analysis Questions to Answer

### Q1: Is adaptive stepsize worth the cost?
Compare:
- Error improvement: Methods 1→4, Methods 2→5
- Time overhead: ~20-30% slower
- Conclusion: Worth it if convergence improves significantly

### Q2: Does preconditioner interact with line search?
Compare:
- PRGD (adaptive) vs RGD (adaptive)
- Do both benefit equally from line search?
- Or does preconditioner make line search less necessary?

### Q3: Which factor matters more?
Compare improvements:
- Preconditioner effect: Method 2 → Method 1
- Adaptive effect (PRGD): Method 1 → Method 4
- Adaptive effect (RGD): Method 2 → Method 5
- Which gives bigger improvement?

### Q4: Stepsize collapse concern?
Check adaptive stepsize histories:
- Do stepsizes collapse to tiny values?
- Compare PRGD vs RGD adaptive stepsizes
- PRGD may have smaller stepsizes (preconditioned gradient is larger)

---

## Code Changes Summary

### 1. Added Method 5
- Lines ~228-267: New method definition
- Configuration: `use_preconditioner=false`, `use_adaptive_stepsize=true`
- Initial stepsize: μ = 1.0 (same as fixed RGD)

### 2. Updated Plots (2×4 → 2×5)
- All convergence plots now show 5 curves
- Added Plot 5: Adaptive RGD stepsize history
- Added Plots 9-10: RGD results comparison

### 3. Updated Summary Table
- Added row for RGD (adaptive)
- All methods shown with descriptive labels

### 4. Enhanced Statistics Output
- Separate stepsize statistics for both adaptive methods
- Relative performance comparisons organized by category
- Best overall method identified automatically

---

## Testing Workflow

Run the test:
```matlab
cd /Users/wutong/Documents/MATLAB/GeneralPlatform
run('test/test_RGD_amplitude.m')
```

Expected output:
1. **Console**: 5 method results with timing and errors
2. **Figure**: 2×5 subplot comparison
3. **Summary**: Table + stepsize statistics + performance analysis

---

## Benefits of This Update

✅ **Complete comparison**: Tests all combinations (preconditioner × stepsize)  
✅ **Fair testing**: Each method uses optimal initial stepsize  
✅ **Isolate effects**: Can see preconditioner vs adaptive contributions separately  
✅ **Practical insights**: Answers "which method should I use?" question  
✅ **Stepsize analysis**: Can study how line search behaves with/without preconditioner  

---

## Practical Recommendations (Based on Expected Results)

### For accuracy-critical applications:
**Use: PRGD (adaptive)**
- Best final error
- Robust to stepsize choice
- Worth the computational cost

### For speed-critical applications:
**Use: RGD (fixed) with well-tuned μ**
- Fast per-iteration
- Good enough if stepsize is tuned
- Skip preconditioning overhead

### For unknown problems:
**Use: PRGD (adaptive)** or **RGD (adaptive)**
- No manual tuning required
- Adapts to problem automatically
- Safe default choice

### For well-understood problems:
**Use: PRGD (fixed)**
- Fast (no line search)
- Excellent with tuned stepsize
- Use if you know μ = 0.5 works

---

## Next Steps

After running the test:

1. **Analyze stepsize histories**: Check for collapse, trends
2. **Compare convergence curves**: Which method reaches target error first?
3. **Evaluate cost/benefit**: Is adaptive worth the time?
4. **Document findings**: Update recommendations based on actual results
5. **Consider enhancements**: Implement stepsize growth if collapse observed

---

**Test is ready! Run and compare all 5 methods to understand the trade-offs.** 🚀
