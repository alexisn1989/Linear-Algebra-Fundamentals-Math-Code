# ADVANCED LINEAR ALGEBRA PROJECT: Inverse, Column Space, Null Space, Nonsquare Matrices
# Scaffolded Version - Fill in the blanks

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("CHAPTER 7 & 8: Advanced Matrix Concepts")
print("=" * 70)

# ============================================================================
# SECTION 1: Inverse Matrices
# ============================================================================

def matrix_inverse(matrix):
    """
    Calculate the inverse of a square matrix.
    
    Math: If A @ A^(-1) = I, then A^(-1) is the inverse of A
    
    TODO: Check determinant first
    Hint: Use np.linalg.det(matrix)
    Hint: Use np.linalg.inv(matrix)
    """
    
    det = np.linalg.det(matrix)  # Calculate determinant
    
    if abs(det) < 1e-10:
        print("WARNING: Matrix is singular (no inverse exists)")
        return None
    
    inverse = np.linalg.inv(matrix)  # Calculate inverse
    return inverse


print("\nTest 1: Invertible Matrix")
A = np.array([[1, 2],
              [3, 4]])
print(f"Matrix A:\n{A}")

A_inv = matrix_inverse(A)
if A_inv is not None:
    print(f"Inverse of A:\n{A_inv}")
    
    # TODO: Verify A @ A^(-1) = I
    identity =A @ A_inv # Multiply A by its inverse
    print(f"A @ A^(-1):\n{np.round(identity, 10)}")
    print(f"(Should be identity matrix)")

# ============================================================================
# SECTION 2: Column Space & Rank
# ============================================================================

def get_rank_and_column_space(matrix):
    """
    Find the rank of a matrix (dimension of column space).
    
    Math: rank = number of linearly independent columns
    
    TODO: Use np.linalg.matrix_rank()
    """
    
    rank = np.linalg.matrix_rank(matrix)  # Use np.linalg.matrix_rank(matrix)
    
    return rank


print("\n" + "=" * 70)
print("Column Space Analysis")
print("=" * 70)

print("\nExample 1: Full Rank")
M1 = np.array([[1, 0],
               [0, 2]])
rank1 = get_rank_and_column_space(M1)
print(f"Matrix:\n{M1}")
print(f"Rank: {rank1} (out of 2 possible)")
print(f"Column space: {2 - rank1} zero, {rank1} linearly independent")

print("\nExample 2: Rank Deficient")
M2 = np.array([[1, 2],
               [1, 2]])  # Columns are identical
rank2 = get_rank_and_column_space(M2)
print(f"Matrix:\n{M2}")
print(f"Rank: {rank2} (out of 2 possible)")
print(f"Column space: Collapses to a line")

# ============================================================================
# SECTION 3: Null Space
# ============================================================================

def find_null_space(matrix):
    """
    Find vectors that map to zero (null space).
    
    Math: All x where A @ x = 0
    
    TODO: Use SVD (Singular Value Decomposition)
    Hint: U, S, Vt = np.linalg.svd(matrix)
    Hint: Check which singular values are ~0
    """
    
    U, S, Vt = np.linalg.svd(matrix)
    
    # TODO: Find null space dimension
    null_threshold = 1e-10
    null_mask =  S < null_threshold  # Compare S < null_threshold
    null_dim = np.sum(null_mask)  # Count how many are true
    
    # Extract null space vectors (last vectors in Vt if null_dim > 0)
    if null_dim > 0:
        null_vecs = Vt[-null_dim:,:].T  # Last null_dim rows of Vt, transposed
    else:
        null_vecs = None
    
    return null_vecs, null_dim


print("\n" + "=" * 70)
print("Null Space Analysis")
print("=" * 70)

print("\nExample 1: Matrix with null space")
A = np.array([[1, 2],
              [0, 0]])  # Second row all zeros
print(f"Matrix:\n{A}")

null_vecs, null_dim = find_null_space(A)
print(f"Null space dimension: {null_dim}")
if null_vecs is not None:
    print(f"Null space basis:\n{null_vecs}")
    
    # TODO: Verify by multiplying
    test_vec = null_vecs[:, 0]
    result = A @ test_vec  # A @ test_vec
    print(f"A @ null_vector = {np.round(result, 10)} (should be ~[0, 0])")

print("\nExample 2: Full rank matrix")
B = np.array([[1, 2],
              [3, 4]])
print(f"Matrix:\n{B}")

null_vecs, null_dim = find_null_space(B)
print(f"Null space dimension: {null_dim}")
print(f"(Only the zero vector)")

# ============================================================================
# SECTION 4: Rank-Nullity Theorem
# ============================================================================

def verify_rank_nullity(matrix):
    """
    Verify: rank(A) + nullity(A) = n
    
    where n = number of columns
    """
    
    rank = np.linalg.matrix_rank(matrix)  # Use np.linalg.matrix_rank(matrix)
    _, nullity = find_null_space(matrix)
    n_cols = matrix.shape[1]
    
    print(f"Rank: {rank}")
    print(f"Nullity: {nullity}")
    print(f"Columns: {n_cols}")
    print(f"Rank + Nullity = {rank} + {nullity} = {rank + nullity}")
    
    # TODO: Check if equal
    if rank + nullity == n_cols:  # rank + nullity == n_cols
        print("✓ Rank-Nullity theorem verified!")
    else:
        print("✗ Something went wrong")


print("\n" + "=" * 70)
print("Rank-Nullity Theorem")
print("=" * 70)

print("\nTest matrix:")
C = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"Matrix (shape {C.shape}):\n{C}")

verify_rank_nullity(C)

# ============================================================================
# SECTION 5: Nonsquare Matrices
# ============================================================================

def transform_nonsquare(input_vector, matrix):
    """
    Apply a nonsquare matrix transformation.
    
    Math: y = A @ x
    where A is m×n (transforms n-dimensional to m-dimensional)
    """
    
    output = matrix @ input_vector # matrix @ input_vector
    return output


print("\n" + "=" * 70)
print("Nonsquare Matrices (Dimension Shifting)")
print("=" * 70)

print("\nExample 1: 3D → 2D (Projection)")
# Project 3D to 2D (drop z-coordinate)
P = np.array([[1, 0, 0],
              [0, 1, 0]])  # 2×3 matrix
print(f"Projection matrix shape: {P.shape}")
print(f"Matrix:\n{P}")

point_3d = np.array([5, 3, 7])
point_2d = transform_nonsquare(point_3d, P)

print(f"\n3D point: {point_3d}")
print(f"After projection: {point_2d}")

print("\nExample 2: 2D → 3D (Embedding)")
# Embed 2D into 3D (add z=0)
E = np.array([[1, 0],
              [0, 1],
              [0, 0]])  # 3×2 matrix
print(f"Embedding matrix shape: {E.shape}")
print(f"Matrix:\n{E}")

point_2d = np.array([3, 4])
point_3d = transform_nonsquare(point_2d, E)

print(f"\n2D point: {point_2d}")
print(f"After embedding: {point_3d}")

# ============================================================================
# CHALLENGES
# ============================================================================

print("\n" + "=" * 70)
print("CHALLENGES (Try These)")
print("=" * 70)

print("""
CHALLENGE 1: Inverse Matrix
Create a 2×2 matrix and:
  a) Calculate its inverse
  
  b) Verify A @ A^(-1) = I
  c) Try with a singular matrix (det=0)
  
  
"
CHALLENGE 2: Null Space
Create a 3×3 rank-deficient matrix (one column = sum of others)
  a) Find its null space
  b) Verify null space vectors map to zero
  c) Check rank-nullity theorem

CHALLENGE 3: Rank-Nullity
Create a 4×2 matrix
  a) Find rank and nullity
  b) Verify rank + nullity = 2

CHALLENGE 4: Nonsquare Transformations
Create a 5×3 matrix and transform random 3D vectors
  a) Test on basis vectors [1,0,0], [0,1,0], [0,0,1]
  b) Visualize the output space (5D is hard, but try 3×2 or 4×2)

CHALLENGE 5: Projection Matrix
A projection matrix P satisfies P @ P = P
Create a projection matrix and verify this property
""")

print("\n" + "=" * 70)
print("Run the guided version to see solutions!")
print("=" * 70)

M = np.array([[2,1],
              [1,3]])
print(f"Matrix M:\n{M}")

M_inv = matrix_inverse(M)
if M_inv is not None:
    print(f"Inverse M:\n{M_inv}")

    result = M @ M_inv
    print(f"\nM @ M^(-1):\n{np.round(result, 10)}")
    print(f"(Should be identity: [[1,0],[0,1]])")

print ("\n" + "-" * 70)
print("Trying singular matrix (det=0")
S = np.array([[1,2],
             [2,4]])

print(f"Singular matrix S:\n{S}")

print(f"Determinant: {np.linalg.det(S): .2e}")

S_inv = matrix_inverse(S)
if S_inv is not None:
    print("✓ Correctly identified singular matrix (no inverse)")



print("\n" + "=" * 70)
print("CHALLENGE 2: Null Space")
print("=" * 70)

M = np.array([[1, 0, 1],
              [0, 1, 1],
              [2, 3, 5]])

print(f"Matrix M (rank-deficient):\n{M}")
print(f"Note: Column 3 = Column 1 + Column 2")

null_vecs, null_dim = find_null_space(M)
print(f"\n find null space{null_dim}")
if null_vecs is not None:
    print(f"Null space basis vectors:\n{null_vecs}")

    print(f"\n Verifying null space vectors:")
    for i in range(null_dim):
        test_vec = null_vecs[:,i]
        result = M @ test_vec
        print(f"M @ null_vec_{i} = {np.round(result, 10)}")
        print(f"(Should be ~[0, 0, 0])")

    print(f"\n" + "-" * 70)
    print("Rank-Nullity Theorem:")
    rank = np.linalg.matrix_rank(M)
    n_cols = M.shape[1]

    print(f"Rank: {rank}")
    print(f"Nullity: {null_dim}")
    print(f"Columns: {n_cols}")
    print(f"Rank + Nullity = {rank} + {null_dim} = {rank + null_dim}")

    if rank + null_dim == n_cols:
        print("✓ Rank-Nullity theorem verified!")
    else:
        print("✗ Something went wrong")

print("\n" + "=" * 70)
print("CHALLENGE 3: Rank-Nullity")
print("=" * 70)

M = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])

print(f"Matrix M (4×2):\n{M}")
print(f"Shape: {M.shape} (4 rows, 2 columns)")

rank = np.linalg.matrix_rank(M)
null_vecs, null_dim = find_null_space(M)

print(f"\nRank: {rank}")
print(f"Nullity: {null_dim}")
print(f"Columns: {M.shape[1]}")

total = rank + null_dim
expected = M.shape[1]

if total == expected:
    print("✓ Rank-Nullity theorem verified!")
else:
    print("✗ Something went wrong")

# Bonus: Interpret what this means
print(f"\nInterpretation:")
print(f"- This 4×2 matrix transforms 2D space to 4D space")
print(f"- Rank {rank} means {rank} linearly independent output directions")
print(f"- Nullity {null_dim} means {null_dim} input directions map to zero")


print("\n" + "=" * 70)
print("CHALLENGE 4: Nonsquare Transformations")
print("=" * 70)

M = np.array([[1, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [2, 0, 1],
              [0, 2, 1]])

print(f"Matrix M (5×3):\n{M}")
print(f"Shape: {M.shape} (transforms 3D → 5D)")

# Fix: Remove extra brackets (should be 1D arrays, not 2D)
basis_x = np.array([1, 0, 0])
basis_y = np.array([0, 1, 0])
basis_z = np.array([0, 0, 1])

result_x = transform_nonsquare(basis_x, M)
result_y = transform_nonsquare(basis_y, M)
result_z = transform_nonsquare(basis_z, M)

print(f"[1, 0, 0] → {result_x}")
print(f"[0, 1, 0] → {result_y}")
print(f"[0, 0, 1] → {result_z}")

print(f"\nTesting random 3D vectors:")
for i in range(3):
    random_vec = np.random.rand(3)
    result = transform_nonsquare(random_vec, M)
    print(f"{np.round(random_vec, 2)} → {np.round(result, 2)}")

# Fix: Move this OUTSIDE the loop and create M_simple
print(f"\n" + "-" * 70)
print("Visualization with simpler 4×2 matrix (2D → 4D):")

M_simple = np.array([[1, 0],
                     [0, 1],
                     [1, 1],
                     [2, 0]])

# Test basis vectors
e1 = np.array([1, 0])
e2 = np.array([0, 1])

out1 = transform_nonsquare(e1, M_simple)
out2 = transform_nonsquare(e2, M_simple)

print(f"[1, 0] → {out1} (column 1 of matrix)")
print(f"[0, 1] → {out2} (column 2 of matrix)")

print(f"\nInterpretation:")
print(f"- Input: 2D space")
print(f"- Output: 4D space (but only spans a 2D subspace of it)")
print(f"- Columns show where basis vectors land")


P = np.array([[1,0],
              [0,0]])

print(f"Projection matrix P (onto x-axis):\n{P}")

P_squared = P @ P
print(f"\nP @ P:\n{P_squared}")
print(f"\nP:\n{P}")

if np.allclose(P_squared, P):
    print("\n✓ P @ P = P (idempotent property verified!)")
else:
    print("\n✗ Something went wrong")

print(f"\n" + "-" * 70)
print("Testing projection on a vector:")

v = np.array([3, 5])
projected = P @ v

print(f"Original vector: {v}")
print(f"After projection: {projected}")
print(f"(Notice y-coordinate dropped to 0)")

print(f"\n" + "-" * 70)
print("Projection matrix (onto line y=x):")

P2 = (1/2) * np.array([[1, 1],
                       [1, 1]])

print(f"Projection matrix P2:\n{P2}")

P2_squared = P @ P2
print(f"\nP @ P:\n{np.round(P2_squared, 10)}")

if np.allclose(P2_squared, P2):
    print("✓ P2 @ P2 = P2 verified!")
else:
    print("✗ Not idempotent")

v2 =  np.array([3, 1])
projected = P @ v2

print(f"\nOriginal vector: {v2}")
print(f"Projected onto y=x: {np.round(projected, 2)}")
print(f"(Now on the line y=x)")
