# Barzilai-Borwein Adaptive Step Size - Quick Reference

## The Simple Idea

Instead of using a fixed step size or complex line search, the **Barzilai-Borwein (BB) method** automatically estimates the best step size using information from the previous iteration.

## The Formula (One Line!)

```
α_t = ||X_t - X_{t-1}||² / <X_t - X_{t-1}, ∇f(X_t) - ∇f(X_{t-1})>
```

That's it! Clip to `[alpha_min, alpha_max]` and you're done.

## Why It Works

**Intuition:** The BB step size approximates the inverse of the local curvature (Hessian).

- **Numerator:** `||s||²` measures how far we moved
- **Denominator:** `<s, y>` measures how much the gradient changed
- **Ratio:** Estimates `1/curvature` ≈ optimal step size

When curvature is high (gradient changes a lot), step size is small.
When curvature is low (gradient changes slowly), step size is large.

## Comparison with Other Methods

| Method | Step Size | Pros | Cons |
|--------|-----------|------|------|
| **Fixed** | Constant α | Simple, no overhead | Requires tuning, not adaptive |
| **Armijo** | Backtrack from α | Guaranteed decrease | Many function evals, complex |
| **BB** | Auto from previous | Simple, efficient, adaptive | Needs 1 previous iterate |

## Implementation (5 lines of code)

```matlab
% After computing new gradient G_t:
if t > 1
    s_norm_sq = sum(sum((X_t - X_prev).^2));
    sy = sum(sum((X_t - X_prev) .* (G_t - G_prev)));
    alpha_t = clip(s_norm_sq / sy, alpha_min, alpha_max);
end

% Simple backtracking if loss doesn't decrease
while f(X_t - alpha_t * D_t) >= f(X_t) && alpha_t > alpha_min
    alpha_t = 0.5 * alpha_t;
end

% Save for next iteration
X_prev = X_t;
G_prev = G_t;
```

## When to Use BB

✅ **Use BB when:**
- You want adaptive step size without tuning
- Problem has varying curvature
- You can afford storing one previous iterate/gradient
- You want simple, proven method

❌ **Don't use BB when:**
- First iteration (no previous data) - use fixed α instead
- Problem is extremely ill-conditioned - may need more robust method
- Memory is extremely tight - BB needs O(n) extra storage

## Typical Parameters

```matlab
params.use_adaptive_stepsize = true;  % Enable BB
params.alpha_min = 1e-10;             % Lower bound
params.alpha_max = 1.0;               % Upper bound  
params.backtrack_beta = 0.5;          % Reduction factor
params.max_linesearch_iter = 20;      % Max backtracks
```

**That's all you need!** No complex tuning of Armijo constants or heuristics.

## Performance Tips

1. **Good initialization matters:** BB uses previous step, so start with reasonable α₀
2. **Watch first few iterations:** BB adapts quickly, expect some backtracks initially
3. **Check `<s,y>` positivity:** Should be positive (negative = bad curvature estimate)
4. **Monitor step sizes:** Should stabilize after initial adaptation period

## Mathematical Background (Optional)

The BB method comes from approximating Newton's equation:
```
H * s = y    (where H is Hessian)
```

Instead of computing full Hessian, BB uses scalar approximation:
```
λ * ||s|| ≈ ||y||    =>    α ≈ 1/λ ≈ ||s||²/<s,y>
```

This gives surprisingly good estimates of local curvature with minimal computation.

## References

**Original Paper:**
- Barzilai, J., & Borwein, J. M. (1988). "Two-point step size gradient methods." IMA Journal of Numerical Analysis, 8(1), 141-148.

**Modern Overview:**
- Raydan, M. (1997). "The Barzilai and Borwein gradient method for the large scale unconstrained minimization problem." SIAM Journal on Optimization, 7(1), 26-33.

**Convergence Analysis:**
- Dai, Y. H., & Fletcher, R. (2005). "Projected Barzilai-Borwein methods for large-scale box-constrained quadratic programming." Numerische Mathematik, 100(1), 21-47.
