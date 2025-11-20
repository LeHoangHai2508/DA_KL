# MFWOA Algorithm 2 & 3 Implementation (Correct per Paper)

## Overview
Corrected implementation of **Multifactorial Whale Optimization Algorithm (MFWOA)** according to:
- **Algorithm 2**: Overall flow with RMP update rule
- **Algorithm 3**: Knowledge transfer mechanisms

## Key Changes from Old to New

### 1. RMP Update Rule (Algorithm 2, Lines 12-15)

**OLD (Incorrect)**:
- RMP was updated every 10 iterations
- Based on success rate of evaluations
- Logic: `rmp = 0.5 * rmp + 0.5 * frac_succ`

**NEW (Per Paper)**:
```python
# EVERY iteration, check if ANY task improved since previous iteration
if g > 0:
    task_improved = any(best_score[t] < best_score_prev[t] for t in range(T))
    if not task_improved:
        # NO task improved -> apply Gaussian perturbation
        delta = 0.1
        gaussian_noise = rng.normal(0, 1)  # N(0,1)
        rmp = np.clip(rmp + delta * gaussian_noise, 0.0, 1.0)
```

**Algorithm 2 Logic**:
- Line 12: IF no best position updated in this iteration
- Line 13: THEN `rmp = rmp + δN(0,1)` with δ = 0.1
- Line 15: ELSE rmp unchanged

### 2. Knowledge Transfer (Algorithm 3)

**OLD (Simplified)**:
- Simple linear interpolation: `new_pos = pop[i] * (1 - β) + leader * β`
- No implementation of two knowledge transfer methods

**NEW (Full Implementation)**:

#### Inter-Task Transfer (Algorithm 3, Lines 2-20):
When `rand_i1 < rmp`:

**First Way** (Lines 5-12, `rand_i2 < 0.2`):
- Traditional WOA mechanics adapted for inter-task
- Uses distance to both current task best AND other task best
- Three modes: Encircling prey, Search for prey, Bubble-net attack
- Formula: `new_pos = leader - A * (D1 + D_other) / 2`
  - `D1`: Distance using current task's best
  - `D_other`: Distance using other task's individual

**Second Way** (Lines 13-18, `rand_i2 ≥ 0.2`):
- Crossover + Mutation approach
- Crossover: `X = (X1 + X2) / 2` (average of solutions)
- Mutation (10% chance): Random perturbation in 1-2 dimensions
- Uses best positions from both tasks

**Skill Factor Change** (Lines 20-21):
- With 50% probability: Change individual's skill factor to other task
- Allows individual to switch task assignment (key for transfer!)

#### Intra-Task Transfer (Algorithm 3, Line 22):
When `rand_i1 ≥ rmp`:
- Standard WOA without knowledge transfer
- Uses current task's best position only

## Code Structure

### Main Loop (mfwoa_multitask function)

```python
for g in range(iters):
    # Step 1: RMP Update (Algorithm 2, lines 12-15)
    if g > 0:
        task_improved = any(best_score[t] < best_score_prev[t] for t in range(T))
        if not task_improved:
            rmp = clip(rmp + 0.1 * N(0,1), 0, 1)
    
    # Step 2: For each individual (Algorithm 3)
    for i in range(pop_size):
        rand_i1 = random()
        if rand_i1 < rmp:  # Inter-task
            # Choose first or second way
            if rand_i2 < 0.2:
                # First way: WOA with D_other
            else:
                # Second way: Crossover + mutation
            # Potentially change skill factor
        else:  # Intra-task
            # Standard WOA
        
        # Evaluate, update best
    
    # Step 3: End iteration
    best_score_prev = best_score.copy()  # Update for next RMP check
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `rmp_init` | 0.3 (default) | Initial cross-task probability (30%) |
| `δ` (delta) | 0.1 | Gaussian noise scale for RMP perturbation |
| `rand_i2 < 0.2` | 0.2 threshold | First way (20% of transfers) |
| `rand_i3 < 0.1` | 0.1 threshold | Mutation rate (10%) |
| `rand_i4 < 0.5` | 0.5 threshold | Skill factor change probability (50%) |

## Advantages of This Implementation

1. **True Adaptive RMP**: RMP self-adjusts based on search progress
   - Increases when stuck (no improvement)
   - Stays stable when improving

2. **Skill Factor Transfer**: Individuals can permanently switch tasks
   - Enables experienced individuals from one task to help another
   - More powerful than simple position sharing

3. **Two Knowledge Transfer Methods**:
   - **First way**: Hybrid WOA leveraging both task's information
   - **Second way**: Direct genetic material exchange (crossover)

4. **Per-Iteration RMP Update**: Faster adaptation vs. every-10-iterations
   - Better response to search dynamics
   - Smoother convergence

## Testing & Validation

Run benchmark with updated MFWOA:
```bash
python -m src.ui.app  # Flask UI (port 5000)
```

Check console output for:
- RMP value each iteration
- Cross-task transfer count
- Mutation operations performed
- Final best FE per task

## References

- **Algorithm 2**: Overall MFWOA flow with RMP update
- **Algorithm 3**: Knowledge transfer in MFWOA
- Paper: "[Your Paper Citation]"
