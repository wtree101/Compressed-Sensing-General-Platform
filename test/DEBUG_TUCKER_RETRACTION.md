# Tucker Retraction Debug Workflow

## Step 1: Run initialization with debug mode to save tensors
```bash
cd test
matlab -batch "test_tucker_lift_init"
```

This will:
- Run Tucker lift initialization with `debug=true`
- Save `T_tucker` and `Grad_F` from iteration 1 to `debug_tucker_tensors.mat`
- Show comparison between retraction and HOSVD in the output

## Step 2: Test saved tensors with original test code
```bash
cd test
matlab -batch "test_tucker_retraction_from_file"
```

This will:
- Load the saved tensors from `debug_tucker_tensors.mat`
- Run the same comparison as `test_tucker_retraction.m`
- Show detailed component analysis
- Visualize differences

## What to look for:

### If retraction test passes but initialization debug shows large error:
1. **Check gradient construction** - The rank-1 case has special handling
2. **Check Up orthogonality** - `U' * Up` should be ~0
3. **Check HOSVD call** - Should use `HOSVD(T, tucker_ranks)` not `HOSVD(T, tucker_ranks*[1,1,1,1])`
4. **Compare cores** - See if G differs significantly

### Expected output:
```
=== Comparison Results ===
Relative error: 1.234e-11
✓ TEST PASSED: Retraction matches HOSVD
```

### If test fails:
The script will show:
- Core tensor differences
- Factor matrix differences (handling sign ambiguity)
- Element-wise error statistics
- Possible reasons for discrepancy

## Files created:
- `debug_tucker_tensors.mat` - Saved tensors from iteration 1
- `test_tucker_retraction_from_file.m` - Test script using saved data
