"""
Hungarian Algorithm for Optimal Bipartite Matching

This module implements the Hungarian algorithm (also known as Munkres algorithm) 
for finding minimum cost perfect matching in bipartite graphs.

Mathematical Background:
-----------------------
The Hungarian algorithm solves the assignment problem:
Given an m x n cost matrix C, find a matching (assignment) that minimizes
the total cost: min Σ C[i, j] for matched pairs (i, j).

The algorithm works by:
1. Subtracting row minima (transforms cost matrix)
2. Finding maximum matching with 0s
3. Covering all 0s with minimum lines
4. Iteratively updating matrix until perfect matching exists

Time Complexity: O(n³)
Space Complexity: O(n²)

References:
-----------
H. Kuhn, "The Hungarian Method for the Assignment Problem",
Naval Research Logistics Quarterly, 2(1-2), 1955, pp. 83-97.

Applications:
- Pattern matching and correlation analysis
- Assignment problems in optimization
- Tracklet association in multi-object tracking

Author: Converted from MATLAB by Python conversion
"""

import numpy as np
from typing import Tuple, Optional


def hungarian(cost_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    r"""
    Hungarian algorithm for optimal assignment problem.
    
    Finds the minimum cost perfect matching in a bipartite graph given
    a cost matrix. Returns both the matching (as binary matrix) and total cost.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (m, n). Entry (i, j) represents the cost
        of assigning row i to column j.
        - Use np.inf to indicate impossible assignments
        - Algorithm minimizes total cost

    Returns
    -------
    matching : np.ndarray
        Binary matrix of same shape as cost_matrix. Entry (i, j) = 1
        indicates assignment of row i to column j, else 0.
        
    total_cost : float
        Minimum total cost of the optimal matching.
        Computed as sum of cost_matrix[matching == 1]

    Notes
    -----
    Algorithm Steps:
    
    1. **Condense**: Remove isolated vertices (rows/columns with all inf)
       to improve efficiency
    
    2. **Ensure Perfect Matching**: If m ≠ n, add virtual vertices
       with maximum cost to make it square
    
    3. **Step 1**: Subtract row minima to create zeros in each row
    
    4. **Step 2**: Find maximum matching by marking independent zeros
    
    5. **Step 3-6**: Iteratively:
       - Cover columns with starred zeros
       - Find uncovered zeros
       - Create alternating paths
       - Update cost matrix
       
    7. **Un-condense**: Map back to original matrix dimensions

    Examples
    --------
    >>> import numpy as np
    >>> from csmf.utils.hungarian import hungarian
    
    # Simple 3x3 cost matrix
    >>> cost = np.array([
    ...     [4, 1, 3],
    ...     [2, 0, 5],
    ...     [3, 2, 2]
    ... ])
    >>> matching, cost = hungarian(cost)
    >>> print(f"Optimal matching (1 = assigned):\\n{matching}")
    >>> print(f"Total cost: {cost}")

    # With impossible assignments
    >>> cost = np.array([
    ...     [1, np.inf, 3],
    ...     [2, 4, 5],
    ...     [np.inf, 1, 2]
    ... ])
    >>> matching, cost = hungarian(cost)

    Raises
    ------
    ValueError
        If cost_matrix is not 2D or has invalid dimensions
    """
    
    if cost_matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got shape {cost_matrix.shape}")
    
    m, n = cost_matrix.shape
    matching = np.zeros_like(cost_matrix)
    
    # Handle empty matrix
    if m == 0 or n == 0:
        return matching, 0.0
    
    # Condense: remove vertices with no edges (all infinite costs)
    # This speeds up the algorithm significantly
    num_connected_cols = np.sum(~np.isinf(cost_matrix), axis=0)  # connections per column
    num_connected_rows = np.sum(~np.isinf(cost_matrix), axis=1)  # connections per row
    
    x_con = np.where(num_connected_rows != 0)[0]  # connected rows
    y_con = np.where(num_connected_cols != 0)[0]  # connected columns
    
    # Assemble condensed cost matrix (square, padded with max cost)
    p_size = max(len(x_con), len(y_con))
    p_cond = np.zeros((p_size, p_size))
    
    # Copy valid costs
    p_cond[:len(x_con), :len(y_con)] = cost_matrix[np.ix_(x_con, y_con)]
    
    if len(x_con) == 0 or len(y_con) == 0:
        return matching, 0.0
    
    # Ensure perfect matching exists by padding with maximum cost
    Edge = p_cond.copy()
    Edge[p_cond != np.inf] = 0
    Edge[p_cond == np.inf] = 1
    
    cnum = _min_line_cover(Edge)
    pmax = np.nanmax(p_cond[~np.isinf(p_cond)])
    p_size = len(p_cond) + cnum
    
    # Pad cost matrix
    p_cond_padded = np.ones((p_size, p_size)) * pmax
    p_cond_padded[:len(x_con), :len(y_con)] = cost_matrix[np.ix_(x_con, y_con)]
    p_cond = p_cond_padded
    
    # Run Hungarian algorithm main loop
    p_cond, r_cov, c_cov, M = _step1(p_cond)
    r_cov, c_cov, M, _ = _step2(p_cond)
    
    stepnum = 3
    exit_flag = True
    
    while exit_flag:
        if stepnum == 3:
            c_cov, stepnum = _step3(M, p_size)
        elif stepnum == 4:
            M, r_cov, c_cov, Z_r, Z_c, stepnum = _step4(p_cond, r_cov, c_cov, M)
        elif stepnum == 5:
            M, r_cov, c_cov, stepnum = _step5(M, Z_r, Z_c, r_cov, c_cov)
        elif stepnum == 6:
            p_cond, stepnum = _step6(p_cond, r_cov, c_cov)
        else:  # stepnum == 7
            exit_flag = False
    
    # Un-condense: map back to original dimensions
    matching[np.ix_(x_con, y_con)] = M[:len(x_con), :len(y_con)]
    
    # Compute total cost (only from valid assignments)
    total_cost = float(np.sum(cost_matrix[matching == 1]))
    
    return matching, total_cost


def _step1(p_cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Step 1: Subtract row minima.
    
    For each row, subtract its minimum value. This creates at least one zero
    in each row while preserving the optimal matching (shifts all costs equally).
    
    .. math::
        C'_{ij} = C_{ij} - \min_j C_{ij}
    """
    p_size = len(p_cond)
    for i in range(p_size):
        row_min = np.min(p_cond[i, :])
        p_cond[i, :] = p_cond[i, :] - row_min
    
    r_cov = np.zeros(p_size, dtype=int)
    c_cov = np.zeros(p_size, dtype=int)
    M = np.zeros((p_size, p_size), dtype=int)
    
    return p_cond, r_cov, c_cov, M


def _step2(p_cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Step 2: Find maximum matching of zeros.
    
    For each zero in the matrix, if its row and column are uncovered,
    mark it as a matched pair (starred zero) and cover the row and column.
    """
    p_size = len(p_cond)
    r_cov = np.zeros(p_size, dtype=int)
    c_cov = np.zeros(p_size, dtype=int)
    M = np.zeros((p_size, p_size), dtype=int)
    
    # Find and mark independent zeros
    for i in range(p_size):
        for j in range(p_size):
            if p_cond[i, j] == 0 and r_cov[i] == 0 and c_cov[j] == 0:
                M[i, j] = 1  # Star this zero
                r_cov[i] = 1
                c_cov[j] = 1
    
    # Reset covers for next iteration
    r_cov = np.zeros(p_size, dtype=int)
    c_cov = np.zeros(p_size, dtype=int)
    
    return r_cov, c_cov, M, p_cond


def _step3(M: np.ndarray, p_size: int) -> Tuple[np.ndarray, int]:
    r"""
    Step 3: Cover columns containing starred zeros.
    
    If all columns are covered, a perfect matching exists (algorithm terminates).
    Otherwise, proceed to find augmenting paths.
    """
    # Cover each column with a starred zero
    c_cov = np.sum(M, axis=0)  # Count starred zeros per column
    
    if np.sum(c_cov) == p_size:
        stepnum = 7  # Perfect matching found
    else:
        stepnum = 4
    
    return c_cov, stepnum


def _step4(
    p_cond: np.ndarray,
    r_cov: np.ndarray,
    c_cov: np.ndarray,
    M: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    r"""
    Step 4: Find uncovered zeros and create augmenting path.
    
    For each uncovered zero:
    - If its row has no starred zero, go to Step 5
    - Otherwise, cover the row and uncover the column of the starred zero
    - Continue until all zeros are covered or augmenting path found
    """
    p_size = len(p_cond)
    Z_r = np.array([], dtype=int)
    Z_c = np.array([], dtype=int)
    zflag = True
    
    while zflag:
        # Find first uncovered zero
        row, col = 0, 0
        found = False
        for i in range(p_size):
            for j in range(p_size):
                if p_cond[i, j] == 0 and r_cov[i] == 0 and c_cov[j] == 0:
                    row, col = i, j
                    found = True
                    break
            if found:
                break
        
        if not found:
            # No uncovered zeros found
            stepnum = 6
            zflag = False
            Z_r = np.array([0], dtype=int)
            Z_c = np.array([0], dtype=int)
        else:
            # Prime the uncovered zero
            M[row, col] = 2
            
            # Check if row has starred zero
            starred_col = np.where(M[row, :] == 1)[0]
            
            if len(starred_col) > 0:
                # Cover row, uncover column with starred zero
                r_cov[row] = 1
                c_cov[starred_col[0]] = 0
            else:
                # No starred zero in row - found augmenting path start
                stepnum = 5
                zflag = False
                Z_r = np.array([row], dtype=int)
                Z_c = np.array([col], dtype=int)
    
    return M, r_cov, c_cov, Z_r, Z_c, stepnum


def _step5(
    M: np.ndarray,
    Z_r: np.ndarray,
    Z_c: np.ndarray,
    r_cov: np.ndarray,
    c_cov: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""
    Step 5: Construct alternating path and flip assignments.
    
    Starting from primed zero Z0, construct alternating path:
    - Z0 (primed) -> Z1 (starred in same column) -> Z2 (primed in same row) -> ...
    
    Continue until reaching a primed zero with no starred zero in its column.
    Then flip all assignments (star->prime, prime->star) along the path.
    """
    ii = 0
    while True:
        # Find starred zero in column of current primed zero
        col = Z_c[ii]
        rindex = np.where(M[:, col] == 1)[0]
        
        if len(rindex) == 0:
            # No starred zero in this column - path complete
            break
        
        # Add starred zero to path
        Z_r = np.append(Z_r, rindex[0])
        Z_c = np.append(Z_c, col)
        ii += 1
        
        # Find primed zero in row of current starred zero
        row = Z_r[ii]
        cindex = np.where(M[row, :] == 2)[0]
        
        if len(cindex) == 0:
            break
        
        # Add primed zero to path
        Z_r = np.append(Z_r, row)
        Z_c = np.append(Z_c, cindex[0])
        ii += 1
    
    # Flip assignments along the path
    for k in range(len(Z_r)):
        if M[Z_r[k], Z_c[k]] == 1:
            M[Z_r[k], Z_c[k]] = 0
        else:
            M[Z_r[k], Z_c[k]] = 1
    
    # Clear covers and remove primes
    r_cov = np.zeros_like(r_cov)
    c_cov = np.zeros_like(c_cov)
    M[M == 2] = 0
    
    return M, r_cov, c_cov, 3


def _step6(
    p_cond: np.ndarray,
    r_cov: np.ndarray,
    c_cov: np.ndarray
) -> Tuple[np.ndarray, int]:
    r"""
    Step 6: Update cost matrix.
    
    Find minimum uncovered value. Subtract from all uncovered columns,
    add to all covered rows. This maintains the optimal matching while
    creating new zeros for the next iteration.
    """
    uncovered_rows = np.where(r_cov == 0)[0]
    uncovered_cols = np.where(c_cov == 0)[0]
    
    if len(uncovered_rows) == 0 or len(uncovered_cols) == 0:
        return p_cond, 4
    
    minval = np.min(p_cond[np.ix_(uncovered_rows, uncovered_cols)])
    
    # Add to covered rows
    p_cond[r_cov == 1, :] += minval
    # Subtract from uncovered columns
    p_cond[:, c_cov == 0] -= minval
    
    return p_cond, 4


def _min_line_cover(Edge: np.ndarray) -> int:
    r"""
    Calculate minimum number of lines needed to cover all zeros.
    
    This determines how many virtual vertices need to be added
    to ensure a perfect matching exists.
    """
    p_size = len(Edge)
    r_cov = np.zeros(p_size, dtype=int)
    c_cov = np.zeros(p_size, dtype=int)
    M = np.zeros((p_size, p_size), dtype=int)
    
    r_cov, c_cov, M, _ = _step2(Edge)
    c_cov, _ = _step3(M, p_size)
    
    M, r_cov, c_cov, Z_r, Z_c, stepnum = _step4(Edge, r_cov, c_cov, M)
    
    if stepnum != 6:
        M, r_cov, c_cov, _ = _step5(M, Z_r, Z_c, r_cov, c_cov)
    
    cnum = p_size - np.sum(r_cov) - np.sum(c_cov)
    return cnum
