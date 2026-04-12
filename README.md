# Linear-Algebra-Fundamentals-Math-Code
Hands-on learning of linear algebra concepts through Python, NumPy, and Matplotlib. This project bridges mathematical theory with practical implementation, focusing on transformations, matrix properties, and geometric intuition.
Why This Project
Linear algebra is the foundation of machine learning, neural networks, and AI. Understanding the math underneath—not just using libraries—separates engineers who build from those who copy.
This project implements core concepts from 3Blue1Brown's Essence of Linear Algebra series to build deep intuition through code.
What You'll Learn
Chapter 1-6: Foundational Concepts

Rotation Matrices — How to rotate vectors in 2D using cos(θ) and sin(θ)
Matrix Multiplication — The @ operator and why it matters
Determinants — Area scaling factors and invertibility
Transformations — Rotating, scaling, shearing vectors in space
Basis Vectors — Why columns of matrices show transformation behavior

Chapter 7-8: Advanced Concepts

Inverse Matrices — Undoing transformations (A @ A^(-1) = I)
Column Space — All possible outputs of a transformation (rank)
Null Space — All vectors that map to zero
Rank-Nullity Theorem — rank + nullity = # of columns
Nonsquare Matrices — Transforming between dimensions (2D → 5D, etc.)
Projection Matrices — Collapsing data onto subspaces (P @ P = P)

Project Structure
linear-algebra-fundamentals/
├── rotation_project_scaffolded.py      # Chapter 1-6: Fill-in-the-blanks version
├── rotation_project_guided.py          # Chapter 1-6: Full explanations + complete code
├── matrices_advanced_scaffolded.py     # Chapter 7-8: Fill-in-the-blanks version
├── matrices_advanced_guided.py         # Chapter 7-8: Full explanations + complete code
└── README.md                           # This file
How to Use
Option 1: Learn by Reading (Guided)
bashpython rotation_project_guided.py
python matrices_advanced_guided.py
Read the code comments. Understand the explanations. Run and observe results.
Option 2: Learn by Doing (Scaffolded)
bashpython rotation_project_scaffolded.py
python matrices_advanced_scaffolded.py
Fill in the blanks (marked with ___). Implement functions. Debug. Understand.
Key Concepts Implemented
Rotation Matrices
pythonM = [ cos(θ)  -sin(θ) ]
    [ sin(θ)   cos(θ) ]

# Rotate [1, 0] by 90°:
point = [1, 0]
rotated = M @ point  # → [0, 1]
Why it works: Columns show where basis vectors [1,0] and [0,1] land.
Determinant (Area Scaling)
pythondet = np.linalg.det(matrix)

# det = 1: area unchanged (rotation, shear)
# det = 4: area scaled 4x (scale 2x in both directions)
# det = 0: matrix is singular (no inverse exists)
Null Space (Vectors Mapping to Zero)
python# For matrix A, find all x where A @ x = 0
null_vecs, null_dim = find_null_space(A)

# Test: A @ null_vector should equal [0, 0, 0, ...]
Nonsquare Matrices (Dimension Shifting)
python# 3D → 2D (projection): drop z-coordinate
P = [[1, 0, 0],
     [0, 1, 0]]

point_3d = [5, 3, 7]
point_2d = P @ point_3d  # → [5, 3]
What's Happening Mathematically
Example: Neural Network Forward Pass
Input (784D):     [pixel1, pixel2, ..., pixel784]
       ↓ multiply by weight matrix (784×128)
Hidden (128D):    [feature1, feature2, ..., feature128]
       ↓ multiply by weight matrix (128×10)
Output (10D):     [prob_0, prob_1, ..., prob_9]
Each layer is a nonsquare matrix transformation, compressing or expanding dimensions.
Example: Coordinate Transformation
Geographic coords (lat, lon) 
       ↓ apply projection matrix
UTM coords (x, y)
       ↓ apply scaling/rotation
Pixel coords on map
Same concept: matrices transform data between spaces.
Challenges & Exercises
Each project includes challenges:
Rotation Project:

Build different transformation matrices (scale, shear, flip)
Compose transformations (rotate, then scale)
Understand determinant as area scaling

Advanced Matrices:

Calculate matrix inverses and verify A @ A^(-1) = I
Find null spaces of rank-deficient matrices
Verify rank-nullity theorem on various matrices
Transform data between dimensions (3D → 5D)
Implement projection matrices and verify idempotence

Running Tests
bash# Run all tests and challenges
python rotation_project_guided.py
python matrices_advanced_guided.py

# Fill in blanks and test your implementation
python rotation_project_scaffolded.py
python matrices_advanced_scaffolded.py
Expected output: All tests pass, all properties verified.
Key Libraries

NumPy — Array operations, matrix multiplication (@), linear algebra functions
Matplotlib — Visualization of transformations before/after

pythonimport numpy as np
import matplotlib.pyplot as plt

# Matrix multiplication
result = matrix @ vector

# Linear algebra functions
det = np.linalg.det(matrix)        # Determinant
inv = np.linalg.inv(matrix)        # Inverse
rank = np.linalg.matrix_rank(m)    # Rank
Concepts That Connect to Your Career
For GeoAI (Geographic + AI)

Coordinate transformations — Converting lat/lon to UTM, pixel coordinates
Spatial indexing — Finding nearby points using transformed spaces
Embeddings — Representing geographic features in high-dimensional space

For Neural Networks

Weights are matrices — Each layer is a nonsquare matrix transformation
Backpropagation — Gradient computation uses matrix multiplication and transposes
Embeddings — Word embeddings, geographic embeddings are vectors in transformed spaces

For Dimension Reduction

PCA (Principal Component Analysis) — Project high-D data to low-D using matrices
Autoencoders — Neural networks that learn compression via nonsquare matrices

Learning Path

Start with rotation matrices — Intuitive, visual, builds foundation
Understand determinants — Connects visualization to numeric properties
Transpose and multiplication — Core operations for all matrix work
Inverse matrices — Undoing transformations
Rank and null space — Understanding matrix structure
Nonsquare matrices — The reality of most real-world transformations
Projection matrices — Practical technique for dimension reduction

Resources

3Blue1Brown — Essence of Linear Algebra — The source for all concepts

Chapter 1-6: Foundational transformations
Chapter 7: Inverse matrices, column space, null space
Chapter 8: Nonsquare matrices


NumPy Documentation — https://numpy.org/doc/
Linear Algebra Visualizations — GeoGebra, Desmos for interactive exploration

Next Steps
After mastering these concepts:

Learn eigenvalues and eigenvectors (Chapter 14 of 3Blue1Brown)
Study matrix decomposition (SVD, QR, Cholesky)
Implement dimensionality reduction (PCA)
Build neural network layers from scratch using matrices
Apply to real data — geographic, image, or text embeddings

Project Goals
✓ Understand linear algebra deeply, not superficially
✓ See the connection between math and code
✓ Build intuition through visualization
✓ Prepare for machine learning and AI engineering
✓ Develop foundation for specialized domains (GeoAI, computer vision, NLP)
Usage in Production
These concepts directly apply to:

Neural network training — Every forward/backward pass uses matrix operations
Data transformation pipelines — Normalizing, projecting, embedding data
Geographic systems — Coordinate transformations, spatial indexing
Compression algorithms — Reducing dimensionality while preserving information
Recommendation systems — Embeddings and similarity calculations

Author
Built as part of intentional learning for GeoAI + Generative AI engineering. The project demonstrates hands-on mastery of foundational mathematics required for advanced AI work.
License
MIT — Use, modify, learn freely.

"If you understand linear algebra, you understand how neural networks actually work."
Start with rotation_project_guided.py. Run the code. Observe the results. Ask "why?" at every step.
