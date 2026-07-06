$\newcommand{\realR}{\mathbb{R}}$
$\newcommand{\complexC}{\mathbb{C}}$
$\newcommand{\integerZ}{\mathbb{Z}}$
$\newcommand{\rationalQ}{\mathbb{Q}}$
$\newcommand{\naturalN}{\mathbb{N}}$

$\newcommand{\bigO}{\mathcal{O}}$

$\newcommand{\LRp}[1]{\left( #1 \right)}$
$\newcommand{\LRs}[1]{\left[ #1 \right]}$
$\newcommand{\LRb}[1]{\left\{ #1 \right\}}$
$\newcommand{\LRn}[1]{\left\| #1 \right\|}$
$\newcommand{\LRt}[1]{\left\langle #1 \right\rangle}$
$\newcommand{\LRfloor}[1]{\left\lfloor #1 \right\rfloor}$
$\newcommand{\LRceil}[1]{\left\lceil #1 \right\rceil}$
$\newcommand{\abs}[1]{\left| #1 \right|}$

$\newcommand{\dint}{\text{d}}$
$\newcommand{\grad}{\nabla}$
$\newcommand{\deriv}[2]{\frac{\dint #1}{\dint #2}}$
$\newcommand{\nderiv}[3]{\frac{\dint #1^{#3}}{\dint^{#3} #2}}$
$\newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}$
$\newcommand{\npderiv}[3]{\frac{\partial^{#3} #1}{\partial #2^{#3}}}$

$\newcommand{\bigand}{\quad \quad \text{and} \quad \quad}$

$\newcommand{\bc}{\bm{c}}$
$\newcommand{\bw}{\bm{w}}$

$\newcommand{\bx}{\bm{x}}$
$\newcommand{\bX}{\bm{X}}$
$\newcommand{\by}{\bm{y}}$
$\newcommand{\bY}{\bm{Y}}$
$\newcommand{\bu}{\bm{u}}$
$\newcommand{\bU}{\bm{U}}$
$\newcommand{\vf}{\bm{f}}$

$\newcommand{\hu}{\hat{u}}$
$\newcommand{\hf}{\hat{f}}$

$\newcommand{\baru}{\bar{u}}$
$\newcommand{\barU}{\bar{U}}$
$\newcommand{\barf}{\bar{f}}$


$\newcommand{\btheta}{\bm{\theta}}$





# SLFA Coding Notes

## Coding Conventions

1. Code ordering:
    1. Min before max

2. Argument ordering:
    1. $\bX$ before $\by$
    2. Monotonicity checks: Extremum before neighbors

3. $D$ matrix indexing: column index -> host point, row index -> neighbors 

    The matrix $D$ is stored in sparse matrix format. Julia uses Compressed Sparse Column (CSC) format for sparse matrices, so to make best use of the CSC format, the column index should correspond to the host point and the corresponding row indices to its neighbors. 
4.  Duplicate points (i.e., $\bx_i = \bx_j$ for $i \ne j$)
    
    1. All the duplicates of a *neighboring point* are added as neighbors
    2. Duplicates of the host point are not taken as neighbors
    3. As a consequence of 4.1 and 4.2, duplicate points will have the same neighbors as one another.
    4. When computing support sets, duplicated points are considered independently of one another. Consequently, it is possible for one duplicate to be in a support set, while another of its duplicates may not.

5. Support sets: 
    1. Support sets are returned as bit vectors
    2. The extrema point and terminal points are included in the support set
    3. Terminal points are returned as a vector of integers

6. Ordering of $\btheta$:
    1. Center
    2. Width(s)
    3. Network weight
    4. Distributed bias

7. $X$ matrix: Columns of X correspond to a single sample. This convention is because Julia is column-major, meaning memory is contiguous within a column of a matrix.

## Computing Neighbors

### Duplicate Points and Neighbors
Occasionally it is useful to have/allow duplicate $\bx$ values. When using noisy data, duplicate samples can be beneficial to help elucidate the true function value at the duplicated point. Even in the absence of noise, different sampling runs may end up sampling the same point, creating duplicates.

**In the presence of noise,** duplicate points *are non-redundant*, regardless of whether they occur in the same sampling set or not. In the presence of noise, duplicate points most likely do not have duplicate values. 

For the sake of the graph/nearest neighbor routines, it can be advantageous to let the vector $\by$ be a represented as a ragged array in the case of duplicates to avoid invoking the cost of nearest neighbor searches multiple times. However, identifying duplicates would require some sort of sorting/searching preprocessing step, which is itself expensive. We could link duplicates in the graph structure itself. In a distance matrix, this would correspond to stored zeros in the sparse matrix. However, uses for self-duplicates are not expected.

**In the absence of noise,** duplicate points in the same sampling set (i.e., not for the quasi-1D case) *are redundant* and can be removed. Duplicate points in different sampling sets are non-redundant for the quasi-1D approach. More importantly, in the absence of noise, **it does not matter which of the duplicates is chosen** since they all have the same value.

**Conclusion:**
Given how the graph structure is used, we will use the convention of duplicate neighbors are neighbors, but duplicates of the host point are not neighbors.

## User-Provided Functions and Parameters

### Parameters

- `N_max`
- `is_monotonic`
- `start_gap`
- `duplicate_tol`
- `k_extrema`
- `T_RBF`
- `conv_thresholds`

### Functions
- `conv_conditions`
- `conv_enforcement`
- `solver(theta0, X, res, A, D, N, T_phi)`
- `score_extrema`
- `get_initial_guess`
