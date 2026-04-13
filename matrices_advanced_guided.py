# ADVANCED LINEAR ALGEBRA PROJECT: Inverse, Column Space, Null Space, Nonsquare Matrices
# Guided Version - Full Code with Explanations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("=" * 70)
print("CHAPTER 7: Inverse Matrices, Column Space, Null Space")
print("=" * 70)

"""
KEY CONCEPTS:

1. INVERSE MATRIX
   If A is a square matrix, A^(-1) is its inverse
   A @ A^(-1) = I (identity matrix)
   
   What it means: Undo a transformation
   If A rotates by 90°, A^(-1) rotates by -90°
   
2. DETERMINANT REVISITED
   det(A) = 0 means A has NO inverse (singular)
   det(A) ≠ 0 means A has an inverse
   
3. COLUMN SPACE
   All possible outputs of the transformation
   "What vectors can this matrix produce?"
   
4. NULL SPACE (Kernel)
   All vectors that map to zero
   "What vectors disappear when transformed?"
   
   Example: [ 1  0 ] @ [x] = [x]
            [ 0  0 ]   [y]   [0]
   
   Null space: any vector [0, y] (y can be anything)
   Because multiplying by 0 makes it disappear
"""

# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: Inverse Matrices")
print("=" * 70)

def matrix_inverse(matrix):
    """Calculate the inverse of a square matrix."""
    det = np.linalg.det(matrix)
    
    if abs(det) < 1e-10:  # Close to zero
        print("WARNING: Determinant is ~0. Matrix is singular (no inverse exists).")
        return None
    
    inverse = np.linalg.inv(matrix)
    return inverse

# Test 1: Invertible matrix
print("\nTest 1: Invertible Matrix")
A = np.array([[1, 2], 
              [3, 4]])
print(f"Matrix A:\n{A}")

A_inv = matrix_inverse(A)
print(f"\nInverse of A:\n{A_inv}")

# Verify: A @ A^(-1) = I
identity_check = A @ A_inv
print(f"\nA @ A^(-1) (should be identity):\n{np.round(identity_check, 10)}")

# Test 2: Non-invertible (singular) matrix
print("\n" + "-" * 70)
print("Test 2: Non-Invertible (Singular) Matrix")
B = np.array([[1, 2],
              [2, 4]])  # Row 2 is 2x Row 1 (linearly dependent)
print(f"Matrix B:\n{B}")
print(f"Determinant: {np.linalg.det(B)}")

B_inv = matrix_inverse(B)
if B_inv is None:
    print("B has no inverse (det = 0)")

# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Column Space")
print("=" * 70)

"""
COLUMN SPACE:
The span of all columns of a matrix.
All possible outputs when you multiply the matrix by any input vector.

For a 2D → 2D transformation:
- Full rank (det ≠ 0): Column space = entire 2D space
- Rank 1: Column space = a line
- Rank 0: Column space = origin (the zero vector)

Rank = number of linearly independent columns
"""

print("\nExample 1: Full Rank (det ≠ 0)")
M1 = np.array([[1, 0],
               [0, 2]])  # Scales: x by 1, y by 2
rank1 = np.linalg.matrix_rank(M1)
det1 = np.linalg.det(M1)
print(f"Matrix:\n{M1}")
print(f"Rank: {rank1} (out of 2 possible)")
print(f"Determinant: {det1}")
print(f"Column space: All of 2D space (full rank)")

print("\nExample 2: Rank 1 (det = 0)")
M2 = np.array([[1, 2],
               [1, 2]])  # Both columns are the same (linearly dependent)
rank2 = np.linalg.matrix_rank(M2)
det2 = np.linalg.det(M2)
print(f"Matrix:\n{M2}")
print(f"Rank: {rank2} (out of 2 possible)")
print(f"Determinant: {det2}")
print(f"Column space: A line (only 1 linearly independent direction)")

print("\nExample 3: Rank 1 (Projection)")
M3 = np.array([[1, 1],
               [0, 0]])  # Projects everything onto x-axis
rank3 = np.linalg.matrix_rank(M3)
det3 = np.linalg.det(M3)
print(f"Matrix:\n{M3}")
print(f"Rank: {rank3}")
print(f"Determinant: {det3}")
print(f"Column space: The x-axis (collapses y dimension)")

# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Null Space")
print("=" * 70)

"""
NULL SPACE (Kernel):
All vectors that map to zero when transformed by the matrix.

For A @ x = 0, what x satisfies this?

Example: A = [ 1  2 ]
            [ 0  0 ]
         
Null space: Solve [ 1  2 ] @ [x] = [0]
                   [ 0  0 ]   [y]   [0]
                   
This gives: x + 2y = 0  →  x = -2y
So null space is all vectors [-2y, y] for any y
Or: y * [-2, 1]  (the line in direction of [-2, 1])
"""

def find_null_space(matrix):
    """Find the null space of a matrix using SVD."""
    # SVD decomposes A = U @ S @ V^T
    # Right singular vectors corresponding to zero singular values = null space
    U, S, Vt = np.linalg.svd(matrix)
    
    # Find which singular values are ~0
    null_space_threshold = 1e-10
    null_mask = S < null_space_threshold
    
    # Number of vectors in null space = number of zero singular values
    null_dim = np.sum(null_mask)
    
    # If there are zero singular values, the last vectors in V are the null space
    if null_dim > 0:
        null_space_vectors = Vt[-null_dim:, :].T
    else:
        null_space_vectors = None
    
    return null_space_vectors, null_dim

print("\nExample 1: Matrix with 1D null space")
A = np.array([[1, 2],
              [0, 0]])  # Row 2 is all zeros
print(f"Matrix A:\n{A}")

null_vecs, null_dim = find_null_space(A)
print(f"Null space dimension: {null_dim}")
if null_vecs is not None:
    print(f"Null space basis vectors:\n{null_vecs}")
    print(f"Interpretation: Any vector proportional to {null_vecs[:, 0]} maps to zero")
    
    # Verify
    test_vector = null_vecs[:, 0]
    result = A @ test_vector
    print(f"A @ {test_vector} = {np.round(result, 10)} (should be ~[0, 0])")

print("\n" + "-" * 70)
print("Example 2: Full rank matrix (null space = {0})")
B = np.array([[1, 2],
              [3, 4]])
print(f"Matrix B:\n{B}")

null_vecs, null_dim = find_null_space(B)
print(f"Null space dimension: {null_dim}")
if null_dim == 0:
    print("Null space contains only the zero vector")

# ============================================================================
print("\n" + "=" * 70)
print("CHAPTER 8: Nonsquare Matrices (Dimension Shifting)")
print("=" * 70)

"""
NONSQUARE MATRICES:
Transform from one dimension to another.

m × n matrix transforms n-dimensional space to m-dimensional space

Examples:
- 2×3 matrix: 3D → 2D (projection, compression)
- 3×2 matrix: 2D → 3D (embedding, expansion)
- m×n where m ≠ n: always has non-trivial null space or can't be inverted
"""

print("\nExample 1: 3D → 2D (Projection)")
# This matrix projects 3D points onto 2D (x-y plane)
P = np.array([[1, 0, 0],
              [0, 1, 0]])  # 2×3: takes [x, y, z] → [x, y]
print(f"Projection matrix (3D → 2D):\n{P}")
print(f"Shape: {P.shape} (2 rows, 3 columns)")

# Test on a 3D point
point_3d = np.array([5, 3, 7])
point_2d = P @ point_3d
print(f"\n3D point: {point_3d}")
print(f"After projection: {point_2d}")
print(f"(Notice z-coordinate is lost)")

print("\nExample 2: 2D → 3D (Embedding)")
# This matrix embeds 2D into 3D
E = np.array([[1, 0],
              [0, 1],
              [0, 0]])  # 3×2: takes [x, y] → [x, y, 0]
print(f"\nEmbedding matrix (2D → 3D):\n{E}")
print(f"Shape: {E.shape} (3 rows, 2 columns)")

point_2d = np.array([3, 4])
point_3d = E @ point_2d
print(f"\n2D point: {point_2d}")
print(f"After embedding: {point_3d}")
print(f"(Added a zero z-coordinate)")

print("\nExample 3: 3×2 matrix visualization")
A = np.array([[2, 0],
              [0, 2],
              [1, 1]])  # 3×2: scales in 2D, then embeds to 3D

print(f"Matrix:\n{A}")
print(f"This transforms 2D vectors to 3D")
print(f"Column 1 [2, 0, 1]: where [1, 0] goes")
print(f"Column 2 [0, 2, 1]: where [0, 1] goes")

# Transform basis vectors
basis_x = np.array([1, 0])
basis_y = np.array([0, 1])

result_x = A @ basis_x
result_y = A @ basis_y

print(f"\n[1, 0] → {result_x}")
print(f"[0, 1] → {result_y}")

# ============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Rank-Nullity Theorem")
print("=" * 70)

"""
RANK-NULLITY THEOREM:
For an m×n matrix A:

rank(A) + nullity(A) = n

Where:
- rank = dimension of column space (how many independent directions)
- nullity = dimension of null space (how many directions map to zero)
- n = number of columns (input dimension)
"""

print("\nExample 1: Full column rank (2×3)")
A = np.array([[1, 0, 0],
              [0, 1, 0]])  # Projects 3D to 2D, losing one dimension
rank = np.linalg.matrix_rank(A)
null_vecs, nullity = find_null_space(A)
n_cols = A.shape[1]

print(f"Matrix shape: {A.shape}")
print(f"Rank: {rank}")
print(f"Nullity: {nullity}")
print(f"Columns: {n_cols}")
print(f"Rank + Nullity = {rank} + {nullity} = {rank + nullity} = {n_cols} ✓")

print("\nExample 2: Rank-deficient (2×3)")
B = np.array([[1, 2, 3],
              [2, 4, 6]])  # Row 2 = 2 × Row 1 (linearly dependent)
rank = np.linalg.matrix_rank(B)
null_vecs, nullity = find_null_space(B)
n_cols = B.shape[1]

print(f"Matrix:\n{B}")
print(f"Matrix shape: {B.shape}")
print(f"Rank: {rank}")
print(f"Nullity: {nullity}")
print(f"Columns: {n_cols}")
print(f"Rank + Nullity = {rank} + {nullity} = {rank + nullity} = {n_cols} ✓")

# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY & CHALLENGES")
print("=" * 70)

print("""
KEY TAKEAWAYS:

1. INVERSE MATRICES
   - Only square matrices can have inverses
   - det ≠ 0 means invertible
   - A @ A^(-1) = I

2. COLUMN SPACE
   - All possible outputs
   - Rank = number of independent columns
   - Full rank (det ≠ 0) = entire output space

3. NULL SPACE
   - All vectors mapping to zero
   - Rank-deficient matrices have non-trivial null space
   - Rank + Nullity = # of columns

4. NONSQUARE MATRICES
   - m×n transforms n-dimensional space to m-dimensional
   - Always rank-deficient if m < n (outputs smaller than inputs)
   - Never invertible (not square)

CHALLENGES:

1. Create an inverse matrix and verify A @ A^(-1) = I
2. Find the null space of a 3×3 rank-deficient matrix
3. Create a 4×2 matrix and visualize its column space
4. Use rank-nullity theorem to verify your results
5. Create a 2×4 nonsquare matrix and test it on random 4D vectors
""")

print("\nReady for scaffolded version? Let's practice!")
