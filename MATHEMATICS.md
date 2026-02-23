# Mathematical Background: Non-negative Matrix Factorization

This document provides detailed mathematical explanations of all algorithms in the CSMF package.

## Table of Contents

1. [NMF Problem Formulation](#nmf-problem-formulation)
2. [Nesterov's Accelerated Gradient Method](#nesterovs-accelerated-gradient-method)
3. [NeNMF Algorithm](#nenmf-algorithm)
4. [Multi-Dataset Extensions](#multi-dataset-extensions)
5. [Optimization Details](#optimization-details)
6. [Convergence Analysis](#convergence-analysis)

---

## NMF Problem Formulation

### Basic NMF Problem

Given a non-negative data matrix $V \in \mathbb{R}^{m \times n}_+$, find non-negative matrices $W \in \mathbb{R}^{m \times r}_+$ and $H \in \mathbb{R}^{r \times n}_+$ that minimize:

$$\min_{W,H} \mathcal{L}(W,H) = \|V - WH\|_F^2 \quad \text{subject to} \quad W,H \geq 0$$

where:
- **V**: Data matrix (m features × n samples)
- **W**: Basis matrix (m features × r factors)
- **H**: Coefficient matrix (r factors × n samples)
- $\|\cdot\|_F$: Frobenius norm

### Frobenius Norm

The Frobenius norm is defined as:

$$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(A^T A)}$$

The reconstruction error can be expanded as:

$$\|V - WH\|_F^2 = \text{tr}((V - WH)^T(V - WH))$$
$$= \text{tr}(V^T V) - 2\text{tr}(W^T V H^T) + \text{tr}(W^T W H H^T)$$
$$= \text{const} - 2\text{tr}(W^T V H^T) + \text{tr}(W^T W H H^T)$$

### KKT Conditions

For the non-negativity constrained problem, the Karush-Kuhn-Tucker (KKT) conditions require:

**For W:**
$$\frac{\partial \mathcal{L}}{\partial W} - \lambda_W = 0$$

where $\lambda_W \geq 0$ and $\lambda_{W,ij} \cdot W_{ij} = 0$ (complementary slackness)

**For H:**
$$\frac{\partial \mathcal{L}}{\partial H} - \lambda_H = 0$$

with $\lambda_H \geq 0$ and $\lambda_{H,ij} \cdot H_{ij} = 0$

---

## Nesterov's Accelerated Gradient Method

### Motivation

Standard gradient descent achieves $O(1/k)$ convergence rate. Nesterov's acceleration achieves:

$$\mathcal{L}(x_k) - \mathcal{L}(x^*) = O(1/k^2)$$

This is a significant improvement for ill-conditioned problems.

### Algorithm Framework

For a convex, differentiable function $f(x)$ with Lipschitz continuous gradient:

**Standard Gradient Descent:**
$$x_{k+1} = x_k - \frac{1}{L}\nabla f(x_k)$$

**Nesterov's Accelerated Gradient:**
$$y_k = x_k - \frac{1}{L}\nabla f(x_k) \quad \text{[Gradient step]}$$
$$x_{k+1} = y_k + \frac{\alpha_k - 1}{\alpha_{k+1}}(y_k - x_k) \quad \text{[Acceleration step]}$$

where the momentum coefficient sequence satisfies:

$$\alpha_0 = 1, \quad \alpha_{k+1} = \frac{1 + \sqrt{1 + 4\alpha_k^2}}{2}$$

### Intuition

The acceleration term $(y_k - x_k)$ "overshoots" based on the previous step, mimicking momentum. The Nesterov sequence ensures:

$$\lim_{k \to \infty} \frac{\alpha_k}{k} = 1$$

providing near-optimal acceleration.

### Non-negative Least Squares (NNLS)

When minimizing a quadratic function subject to $x \geq 0$:

$$\min_x \|Ax - b\|^2 \quad \text{subject to} \quad x \geq 0$$

The accelerated method becomes:

$$y_k = \max(x_k - \frac{1}{L}\nabla f(x_k), 0) \quad \text{[Projected gradient step]}$$
$$x_{k+1} = y_k + \frac{\alpha_k - 1}{\alpha_{k+1}}(y_k - x_k)$$

The projection onto $\mathbb{R}^n_+$ preserves the convergence guarantees.

---

## NeNMF Algorithm

### Problem

Factorize matrix $V$ using alternating optimization for $W$ and $H$.

### Gradient Computation

**Gradient with respect to H:**
$$\nabla_H \mathcal{L} = W^T(WH - V) = W^T WH - W^T V$$

Define:
- $G_H = W^T WH - W^T V$ (gradient)
- $\Sigma_W = W^T W$ (Gram matrix)
- $P_H = W^T V$ (data projection)

Then: $G_H = \Sigma_W H - P_H$

**Gradient with respect to W:**
$$\nabla_W \mathcal{L} = (WH - V)H^T = WHH^T - VH^T$$

Define:
- $G_W = WHH^T - VH^T$ (gradient)
- $\Sigma_H = HH^T$ (Gram matrix)
- $P_W = VH^T$ (data projection)

Then: $G_W = W\Sigma_H - P_W$

### Lipschitz Constant

The Lipschitz constant for the gradient is the largest eigenvalue:

$$L_H = \|\Sigma_W\|_2 = \lambda_{\max}(W^T W)$$
$$L_W = \|\Sigma_H\|_2 = \lambda_{\max}(HH^T)$$

In Python, this is efficiently computed as:
```python
L = np.linalg.norm(WtW)  # For computing L_H
```

### NeNMF Pseudocode

```
Input: V (m×n), r (target rank), parameters
Initialize W, H randomly
Compute WtW, WtV, HHt, HVt

for iter = 1 to MaxIter:
    # Optimize H with W fixed
    [H, iterH] ← NNLS(H, WtW, WtV, ...)
    Update HHt, HVt
    
    # Optimize W with H fixed
    [W, iterW] ← NNLS(W^T, HHt, HVt, ...)
    W ← W^T
    Update WtW, WtV
    
    # Check convergence
    δ ← ComputeStoppingCriterion(W, H, ∇W, ∇H)
    if δ ≤ tol × init_δ:
        break

return W, H, iter, elapsed_time
```

### Convergence Properties

**Theorem (Nesterov 1983 + NNLS adaptation):**

For the non-negative least squares subproblems, using Nesterov acceleration:

$$\|x_k - x^*\|_2 \leq \frac{4L\|x_0 - x^*\|_2^2}{(k+1)^2}$$

This O(1/k²) rate applies to each inner optimization.

---

## Multi-Dataset Extensions

### CSMF: Common and Specific Matrix Factorization

#### Problem Formulation

Given K datasets $\{X^{(k)}\}_{k=1}^K$, decompose each as:

$$X^{(k)} \approx W^c H^{c,k} + W^{s,k} H^{s,k}$$

Objective:

$$\min_{W^c,W^{s,k},H^{c,k},H^{s,k}} \sum_{k=1}^K \|X^{(k)} - W^c H^{c,k} - W^{s,k} H^{s,k}\|_F^2$$

subject to all factors $\geq 0$.

where:
- $W^c \in \mathbb{R}^{m \times r_c}$: Common basis
- $W^{s,k} \in \mathbb{R}^{m \times r_{s,k}}$: Specific basis for dataset k
- $H^{c,k} \in \mathbb{R}^{r_c \times n_k}$: Common coefficients
- $H^{s,k} \in \mathbb{R}^{r_{s,k} \times n_k}$: Specific coefficients

#### Alternating Optimization

**Step 1: Update Common Components**

Compute residuals: $R^{(k)} = X^{(k)} - W^{s,k} H^{s,k}$

Concatenate: $\mathcal{R} = [R^{(1)} | R^{(2)} | \cdots | R^{(K)}]$

Factorize: $\mathcal{R} \approx W^c [\![ H^{c,1}, H^{c,2}, \ldots, H^{c,K} ]\!]$

where $[\![\cdot]\!]$ denotes horizontal concatenation.

**Step 2: Update Specific Components**

For each dataset k:

Compute residual: $R^{(k)} = X^{(k)} - W^c H^{c,k}$

Factorize: $R^{(k)} \approx W^{s,k} H^{s,k}$

**Step 3: Normalize**

After assembly, normalize for numerical stability:

$$W \leftarrow W \text{diag}(s)^{-1}, \quad H \leftarrow \text{diag}(s) H$$

where $s_j = \sum_i W_{ij}$ (column sums).

#### Convergence Criterion for CSMF

The overall convergence is monitored by:

$$\delta = \|\text{proj-grad}(W, H)\|_2$$

computed on the concatenated factors.

### iNMF: Integrative NMF

#### Algorithm

1. **Independent Factorization:** 
   For each dataset k, factorize:
   $$X^{(k)} \approx W^{(k)} H^{(k)}$$
   
   using rank $r_k = r_c + r_{s,k}$ (combined rank)

2. **Correlation Analysis:**
   Compute pairwise column correlations:
   $$C_{ij} = \frac{(W^{(1)})_i \cdot (W^{(2)})_j}{\|(W^{(1)})_i\|_2 \|(W^{(2)})_j\|_2}$$

3. **Optimal Matching:**
   Create distance matrix: $D = 1 - C$
   
   Use Hungarian algorithm (O(n³)) to find optimal matching minimizing:
   $$\min_{\pi} \sum_i D_{i,\pi(i)}$$

4. **Pattern Selection:**
   Select top $r_c$ matched pairs as common basis
   
   Average matched vectors: $W^c_{:,i} = \frac{W^{(1)}_{:,\sigma(i)} + W^{(2)}_{:,\pi(\sigma(i))}}{2}$
   
   Unmatched vectors form specific basis

#### Hungarian Algorithm Details

The Hungarian algorithm solves the linear assignment problem in O(n³) time:

**Key Steps:**
1. Row reduction: subtract row minimum from each row
2. Find maximum independent zero set (initial matching)
3. If matching complete → optimal solution
4. Otherwise: construct alternating paths and update cost matrix
5. Iterate until perfect matching found

This is more efficient than brute force O(n!) enumeration.

### jNMF: Joint NMF with Thresholding

#### Problem Formulation

Given K datasets, factorize jointly:

$$[X^{(1)} | X^{(2)} | \cdots | X^{(K)}] \approx W^c [\![ H^{(1)} | H^{(2)} | \cdots | H^{(K)} ]\!]$$

Then threshold to identify significant components, and factorize residuals.

#### Thresholding with Z-Scores

For matrix element $X_{ij}$, compute z-score:

$$z_{ij} = \frac{X_{ij} - \text{mean}(X)}{\text{std}(X)}$$

Apply threshold:

$$X'_{ij} = \begin{cases} X_{ij} & \text{if } |z_{ij}| > \text{cut} \\ 0 & \text{otherwise} \end{cases}$$

This identifies "significant" elements while removing noise.

---

## Optimization Details

### Stopping Criteria

#### Criterion 1: Projected Gradient Norm (default)

For variable $X \geq 0$ with gradient $\nabla f(X)$:

$$\text{PG} = \begin{cases} 
\nabla f(X)_{ij} & \text{if } X_{ij} > 0 \\
\min(\nabla f(X)_{ij}, 0) & \text{if } X_{ij} = 0
\end{cases}$$

Stopping criterion:

$$\|\text{PG}\|_2 \leq \epsilon$$

#### Criterion 2: Normalized Projected Gradient

$$\frac{\|\text{PG}\|_2}{|\text{supp}(\text{PG})|}$$

where $|\text{supp}(\cdot)|$ is the support size.

#### Criterion 3: Normalized KKT Residual

$$\frac{\|\min(X, \nabla f(X))\|_1}{|\{(i,j) : |\min(X, \nabla f(X))_{ij}| > \epsilon\}|}$$

This measures KKT violation: $\min(X, \nabla f) \approx 0$ at optimality.

### Tolerance Adaptation

In NeNMF, inner NNLS tolerances are adapted:

```
if num_inner_iterations ≤ min_iter:
    tolerance ← tolerance / 10  # Tighten tolerance
```

This ensures we spend more iterations initially, then focus on difficult subproblems.

---

## Convergence Analysis

### Theoretical Guarantees

**Theorem (Convergence of Alternating Nesterov Methods):**

For non-convex NMF with alternating Nesterov optimization:

1. Every limit point satisfies KKT conditions (for critical points)
2. Iteration sequence is bounded
3. Objective function is non-increasing

Proof sketch:
- Each NeNMF step decreases objective: $\mathcal{L}(W^{t+1}, H^t) \leq \mathcal{L}(W^t, H^t)$
- Alternating optimization: $\mathcal{L}(W^{t+1}, H^{t+1}) \leq \mathcal{L}(W^{t+1}, H^t)$
- Monotone sequence bounded below by 0 converges

### Practical Convergence

In practice, we observe:

1. **Fast initial decrease:** First 10-50 iterations drop error significantly
2. **Slow tail convergence:** Last iterations show marginal improvement
3. **Sensitivity to initialization:** Different random seeds give different local minima
4. **Early stopping possible:** Stop when relative improvement < tolerance

---

## References

### Core Papers

1. Guan, N., Tao, D., Luo, Z., & Yuan, B. (2012). "NeNMF: An Optimal Gradient Method for Non-negative Matrix Factorization." IEEE TSP, 60(6), 2882-2898.

2. Nesterov, Y. (1983). "A method of solving a convex programming problem with convergence rate O(1/k²)." Soviet Mathematics Doklady, 27(2), 372-376.

3. Lin, C.-J. (2007). "Projected gradient methods for nonnegative matrix factorization." Neural Computation, 19(10), 2756-2779.

### Related Works

4. Lee, D. D., & Seung, H. S. (1999). "Learning the parts of objects by non-negative matrix factorization." Nature, 401(6755), 788-791.

5. Kuhn, H. W. (1955). "The Hungarian method for the assignment problem." Naval Research Logistics Quarterly, 2(1-2), 83-97.

6. Boyd, S., & Parikh, N. (2014). "Proximal Algorithms." Foundations and Trends in ML, 1(3), 123-231.

---

## Appendix: Numerical Stability Tips

### Preventing Overflow/Underflow

1. **Normalize W:** Column sums should be ~1
   ```python
   W = W / np.sum(W, axis=0, keepdims=True)
   ```

2. **Monitor matrix condition:** $\text{cond}(W^T W) = \lambda_{\max} / \lambda_{\min}$
   ```python
   eigs = np.linalg.eigvalsh(W.T @ W)
   cond_number = eigs[-1] / eigs[0]
   ```

3. **Use double precision:** Always use `float64`, not `float32`

4. **Add small epsilon:** Prevent zero gradients
   ```python
   H = np.maximum(H, eps)  # eps ~ 1e-10
   ```

### Debugging Convergence Issues

Track these quantities:

1. **Gradient norm:** Should decrease monotonically
2. **Relative change:** $|(f_{k} - f_{k-1})/f_k|$
3. **KKT violation:** $\|\max(0, -\nabla f(X)) / X\|_{\max}$

---

**Last Updated:** 2026

For more details on specific functions, see the docstrings in source code.
