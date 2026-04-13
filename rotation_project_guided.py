# MATH + CODE PROJECT: Coordinate Transformation Visualizer
# Guided Version - Full Code with Explanations

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("LESSON 1: Understanding Rotation Matrices")
print("=" * 70)

"""
MATH CONCEPT:
A rotation matrix by angle θ counterclockwise is:

M = [ cos(θ)  -sin(θ) ]
    [ sin(θ)   cos(θ) ]

WHY? Because it tells us where the basis vectors go:
- [1, 0] becomes [cos(θ), sin(θ)]  (red vector rotates)
- [0, 1] becomes [-sin(θ), cos(θ)] (blue vector rotates)

EXAMPLE: 90° rotation
M = [ cos(90°)  -sin(90°) ]  =  [ 0  -1 ]
    [ sin(90°)   cos(90°) ]     [ 1   0 ]

This matches what you learned today!
"""

def rotation_matrix(angle_degrees):
    """Create a 2D rotation matrix for a given angle in degrees."""
    angle_rad = np.radians(angle_degrees)
    
    matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    return matrix


# Test it
M_90 = rotation_matrix(90)
print("\n90° Rotation Matrix:")
print(M_90)
print("(Notice the values match [ 0 -1 ] / [ 1 0 ])")

# ============================================================================
print("\n" + "=" * 70)
print("LESSON 2: Apply Matrix to a Single Point")
print("=" * 70)

"""
MATH CONCEPT:
To transform a point [x, y] by matrix M:

result = M × [x, y] = [a*x + b*y, c*x + d*y]

In code: use the @ operator (matrix multiplication)
"""

def transform_point(point, matrix):
    """Apply a transformation matrix to a single point."""
    # point is [x, y], matrix is 2x2
    result = matrix @ point
    return result


# Test it
point = np.array([1.0, 0.0])  # Point on the x-axis
rotated = transform_point(point, M_90)

print(f"\nOriginal point: {point}")
print(f"After 90° rotation: {rotated}")
print(f"Expected: approximately [0, 1] (pointing up)")
print(f"Correct? {np.allclose(rotated, [0, 1])}")

# ============================================================================
print("\n" + "=" * 70)
print("LESSON 3: Transform Multiple Points at Once (Batch Operation)")
print("=" * 70)

"""
MATH CONCEPT:
You can transform multiple points with one operation.
If you have n points as a matrix (n rows, 2 columns):

points = [[x1, y1],
          [x2, y2],
          [x3, y3]]

Transform all: M @ points.T (transpose needed for dimensions to work)
Then transpose back to get (n, 2) shape.
"""

def transform_points(points, matrix):
    """Apply transformation to multiple points at once."""
    # points.T converts (n, 2) to (2, n) for matrix multiplication
    # Result is (2, n), so transpose back to (n, 2)
    transformed = (matrix @ points.T).T
    return transformed


# Create a square
square = np.array([
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]
])

rotated_square = transform_points(square, M_90)

print("\nOriginal square corners:")
print(square)
print("\nAfter 90° rotation:")
print(rotated_square)
print("\nNotice: each point rotated 90° counterclockwise")

# ============================================================================
print("\n" + "=" * 70)
print("LESSON 4: Transformation Properties - Determinant & Trace")
print("=" * 70)

"""
MATH CONCEPTS:

Determinant (det):
- Tells you how much the transformation scales AREA
- det = 1: area preserved
- det = 2: area doubled
- det = 0: space collapsed to a line (bad!)

Trace:
- Sum of diagonal elements (a + d)
- Less intuitive, but useful in advanced linear algebra
"""

def get_transformation_properties(matrix):
    """Extract determinant, trace, and other properties."""
    det = np.linalg.det(matrix)
    trace = np.trace(matrix)
    
    return {
        'determinant': det,
        'trace': trace,
        'area_scale_factor': abs(det)
    }


# Test on different transformations
print("\nRotation 90°:")
props = get_transformation_properties(M_90)
print(f"  Determinant: {props['determinant']:.4f}")
print(f"  Trace: {props['trace']:.4f}")
print(f"  Area scaling: {props['area_scale_factor']:.4f}x")
print(f"  Interpretation: Rotation preserves area ✓")

print("\nScale 2x ([ 2  0 ] / [ 0  2 ]):")
M_scale = np.array([[2, 0], [0, 2]])
props = get_transformation_properties(M_scale)
print(f"  Determinant: {props['determinant']:.4f}")
print(f"  Area scaling: {props['area_scale_factor']:.4f}x")
print(f"  Interpretation: Area quadrupled (2×2=4) ✓")

print("\nShear ([ 1  0.5 ] / [ 0  1 ]):")
M_shear = np.array([[1, 0.5], [0, 1]])
props = get_transformation_properties(M_shear)
print(f"  Determinant: {props['determinant']:.4f}")
print(f"  Area scaling: {props['area_scale_factor']:.4f}x")
print(f"  Interpretation: Area preserved, but shape changed ✓")

# ============================================================================
print("\n" + "=" * 70)
print("LESSON 5: Visualize Transformations")
print("=" * 70)

def visualize_transformation(points, matrix, title="Transformation", angle=None):
    """Plot original and transformed points side by side."""
    transformed_points = transform_points(points, matrix)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original points
    ax1.scatter(points[:, 0], points[:, 1], color='red', s=100, label='Points', zorder=3)
    
    # Draw square outline
    square_closed = np.vstack([points, points[0]])  # Close the square
    ax1.plot(square_closed[:, 0], square_closed[:, 1], 'r-', alpha=0.5, linewidth=2)
    
    # Draw basis vectors
    ax1.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='darkred', ec='darkred', alpha=0.7)
    ax1.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='darkblue', ec='darkblue', alpha=0.7)
    ax1.text(1.15, 0, '[1, 0]', fontsize=10)
    ax1.text(0, 1.15, '[0, 1]', fontsize=10)
    
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Original Points', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Transformed points
    ax2.scatter(transformed_points[:, 0], transformed_points[:, 1], color='blue', s=100, label='Points', zorder=3)
    
    # Draw transformed square outline
    transformed_closed = np.vstack([transformed_points, transformed_points[0]])
    ax2.plot(transformed_closed[:, 0], transformed_closed[:, 1], 'b-', alpha=0.5, linewidth=2)
    
    # Draw transformed basis vectors
    basis_x_transformed = matrix @ np.array([1, 0])
    basis_y_transformed = matrix @ np.array([0, 1])
    
    ax2.arrow(0, 0, basis_x_transformed[0], basis_x_transformed[1], 
              head_width=0.1, head_length=0.1, fc='darkred', ec='darkred', alpha=0.7)
    ax2.arrow(0, 0, basis_y_transformed[0], basis_y_transformed[1], 
              head_width=0.1, head_length=0.1, fc='darkblue', ec='darkblue', alpha=0.7)
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title(f'After {title}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    return fig


# Uncomment to visualize:
# visualize_transformation(square, M_90, "90° Rotation")
# plt.show()

print("\n✓ Visualization function created")
print("  Uncomment the lines above to see plots")

# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENTS: Try Different Transformations")
print("=" * 70)

print("\nExperiment 1: Different rotation angles")
for angle in [30, 45, 90, 180]:
    M = rotation_matrix(angle)
    props = get_transformation_properties(M)
    print(f"  {angle:3d}°: det={props['determinant']:6.3f}, trace={props['trace']:6.3f}")

print("\nExperiment 2: Composition - What if you rotate THEN scale?")
M_rotate = rotation_matrix(45)
M_scale = np.array([[2, 0], [0, 2]])

# Compose: first rotate, then scale
M_composed = M_scale @ M_rotate

point = np.array([1, 0])
result = M_composed @ point

print(f"  Original point: {point}")
print(f"  After: rotate 45° THEN scale 2x")
print(f"  Result: {result}")

print("\nExperiment 3: Does order matter?")
M_composed_reverse = M_rotate @ M_scale
result_reverse = M_composed_reverse @ point
print(f"  Rotate first: {result}")
print(f"  Scale first: {result_reverse}")
print(f"  Same? {np.allclose(result, result_reverse)}")
print(f"  (Answer: Order DOES matter! Matrix multiplication is not commutative)")

# ============================================================================
print("\n" + "=" * 70)
print("CHALLENGES FOR YOU TO TRY")
print("=" * 70)

print("""
1. SCALE MATRIX
   Create: scale_matrix(sx, sy)
   Should return: [[ sx,  0 ], [ 0, sy ]]
   Test: Apply to square, visualize

2. SHEAR MATRIX  
   Create: shear_matrix(shear_amount)
   Should return: [[ 1, shear ], [ 0, 1 ]]
   Test: Apply to square, see it slant

3. DETERMINANT CHALLENGE
   - Create a 3x scale matrix [[ 3, 0 ], [ 0, 3 ]]
   - Calculate determinant
   - What's the area scaling? (should be 9)
   - Why? (because 3 × 3 = 9)

4. COMPOSITION CHALLENGE
   - Rotate by 30°
   - Then shear by 0.5
   - Apply to square and visualize
   - Then reverse order and compare

5. FLIP MATRIX
   Create: flip_x_matrix() = [[ -1, 0 ], [ 0, 1 ]]
   Test: What's the determinant? What does negative det mean?
   (Hint: orientation flip!)
""")

print("\n" + "=" * 70)
print("Ready to experiment? Uncomment visualizations and try the challenges!")
print("=" * 70)
