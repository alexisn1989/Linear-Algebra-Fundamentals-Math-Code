# MATH + CODE PROJECT: Coordinate Transformation Visualizer
# Scaffolded Version - Fill in the blanks

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# LESSON 1: Build a 2D Rotation Matrix
# ============================================================================

def rotation_matrix(angle_degrees):
    """
    Create a rotation matrix for a given angle.
    
    Math: To rotate counterclockwise by θ:
    M = [ cos(θ)  -sin(θ) ]
        [ sin(θ)   cos(θ) ]
    
    TODO: Fill in the matrix values
    Hint: Use np.radians() to convert degrees to radians
    Hint: Use np.cos() and np.sin()
    """
    angle_rad = np.radians(angle_degrees)
    
    # FILL IN THIS MATRIX
    matrix = np.array([
        [___,  ___],      # Row 1: [cos(θ), -sin(θ)]
        [___,  ___]       # Row 2: [sin(θ), cos(θ)]
    ])
    
    return matrix


# ============================================================================
# LESSON 2: Apply Rotation to a Coordinate
# ============================================================================

def transform_point(point, matrix):
    """
    Apply a transformation matrix to a single point.
    
    Math: result = matrix × point
    [ a  b ] [ x ]   =  [ ax + by ]
    [ c  d ] [ y ]      [ cx + dy ]
    
    TODO: Implement matrix multiplication using @ operator
    Hint: matrix @ point should give you the result
    """
    
    result = ___  # Use @ operator for matrix multiplication
    return result


# ============================================================================
# LESSON 3: Transform Multiple Points (Batch)
# ============================================================================

def transform_points(points, matrix):
    """
    Apply transformation to multiple points at once.
    
    points: array of shape (n, 2) - n points with [x, y] coordinates
    matrix: 2x2 transformation matrix
    
    TODO: Apply the matrix to all points
    Hint: matrix @ points.T should work (transpose needed)
    Hint: Then transpose back to get shape (n, 2)
    """
    
    transformed = ___  # Apply matrix to points
    return transformed


# ============================================================================
# LESSON 4: Calculate Properties of the Transformation
# ============================================================================

def get_transformation_properties(matrix):
    """
    Extract important info about the transformation.
    
    TODO: Calculate determinant and trace
    Hint: np.linalg.det(matrix) gives determinant
    Hint: np.trace(matrix) gives trace (sum of diagonal)
    """
    
    det = ___  # Determinant
    trace = ___  # Trace
    
    return {
        'determinant': det,
        'trace': trace,
        'area_scale': abs(det)  # How much area scaled
    }


# ============================================================================
# LESSON 5: Visualize Before & After
# ============================================================================

def visualize_transformation(points, matrix, title="Transformation"):
    """
    Plot original points and transformed points side by side.
    
    TODO: Transform the points and create two plots
    """
    
    # Transform points
    transformed_points = transform_points(points, matrix)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original points
    ax1.scatter(points[:, 0], points[:, 1], color='red', s=100, label='Original')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Original Points')
    ax1.legend()
    
    # Transformed points
    # TODO: Plot transformed_points on ax2 with color 'blue'
    # Hint: Use ax2.scatter() like the example above
    
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title(f'After {title}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN: Test Your Implementation
# ============================================================================

if __name__ == "__main__":
    
    # Test 1: Create a rotation matrix for 90 degrees
    print("=" * 60)
    print("TEST 1: Rotation Matrix (90 degrees)")
    print("=" * 60)
    
    M = rotation_matrix(90)
    print("Matrix:\n", M)
    
    props = get_transformation_properties(M)
    print(f"Determinant: {props['determinant']:.4f}")
    print(f"Trace: {props['trace']:.4f}")
    print()
    
    # Test 2: Rotate a single point
    print("=" * 60)
    print("TEST 2: Transform Single Point")
    print("=" * 60)
    
    point = np.array([1, 0])
    rotated = transform_point(point, M)
    print(f"Original point: {point}")
    print(f"After 90° rotation: {rotated}")
    print(f"Expected: [~0, ~1]")
    print()
    
    # Test 3: Rotate multiple points
    print("=" * 60)
    print("TEST 3: Transform Multiple Points")
    print("=" * 60)
    
    # Create a square
    points = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0]
    ])
    
    rotated_points = transform_points(points, M)
    print("Original points:\n", points)
    print("\nRotated points:\n", rotated_points)
    
    # TODO: Uncomment below to visualize
    # visualize_transformation(points, M, "90° Rotation")
    
    # Test 4: Try different rotations
    print("=" * 60)
    print("TEST 4: Try Different Angles")
    print("=" * 60)
    
    for angle in [45, 90, 180]:
        M = rotation_matrix(angle)
        props = get_transformation_properties(M)
        print(f"Angle: {angle}° | Det: {props['determinant']:.2f} | Trace: {props['trace']:.2f}")


# ============================================================================
# CHALLENGES (After completing above)
# ============================================================================

"""
CHALLENGE 1: Scale Matrix
Create a function scale_matrix(sx, sy) that returns:
[ sx   0 ]
[  0  sy ]

Then test it on your points.

CHALLENGE 2: Compose Transformations
Create a function compose_matrices(M1, M2) that multiplies two matrices.
Then rotate by 45°, THEN scale by 2x.
Does order matter? Try M1 @ M2 vs M2 @ M1.

CHALLENGE 3: Determinant Interpretation
Create a scale matrix with sx=2, sy=2.
What's the determinant? (hint: should be 4)
Why? (hint: area scaling)

CHALLENGE 4: Shear Matrix
Create a shear_matrix(shear_amount) that returns:
[ 1  shear ]
[ 0    1   ]
Apply it to a square. Watch how it slants.
"""
