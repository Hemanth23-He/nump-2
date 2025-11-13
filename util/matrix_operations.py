"""A module to perform matrix operations - NumPy optimized version.
"""
import numpy as np
from math import log


def matrix_vector_multiply(mat, vec):
    """Multiplies a matrix by a vector.
    
    Multiplies an m x n matrix by an n x 1 vector (represented
    as a list).
    
    Args:
        mat (2-D list): Matrix to multiply.
        vec (list): Vector to multiply.
    
    Returns:
        Product of mat and vec (an m x 1 vector) as a list
    """
    # Convert to numpy arrays if needed
    if not isinstance(mat, np.ndarray):
        mat = np.array(mat)
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    
    # Vectorized matrix-vector multiplication
    prod = np.dot(mat, vec)
    
    # Return as list for compatibility (or keep as array - both work)
    return prod.tolist() if isinstance(prod, np.ndarray) else prod


def add(vec1, vec2):
    """Adds two vectors.
    
    Adds a length-n list to another length-n list.
    
    Args:
        vec1 (list): First vector.
        vec2 (list): Second vector.
    
    Returns:
        Sum of vec1 and vec2.
    """
    # Convert to numpy arrays if needed
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2)
    
    assert len(vec1) == len(vec2)
    
    # Vectorized addition
    result = vec1 + vec2
    
    return result.tolist() if isinstance(result, np.ndarray) else result


def scalar_multiply(vec, constant):
    """Multiplies a scalar by a vector.
    
    Multiplies a vector by a scalar.
    
    Args:
        vec (list): Vector to multiply.
        constant (float): Scalar to multiply.
    
    Returns:
        Product of vec and constant.
    """
    # Convert to numpy array if needed
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    
    # Vectorized scalar multiplication
    result = vec * constant
    
    return result.tolist() if isinstance(result, np.ndarray) else result


def diagonal(mat, diag_index):
    """Returns ith diagonal of matrix, where i is the diag_index.
    
    Returns the ith diagonal (A_0i, A_1(i+1), ..., A_N(i-1)) of a matrix A,
    where i is the diag_index.
    
    Args:
        mat (2-D list): Matrix.
        diag_index (int): Index of diagonal to return.
    
    Returns:
        Diagonal of a matrix.
    """
    # Convert to numpy array if needed
    if not isinstance(mat, np.ndarray):
        mat = np.array(mat)
    
    n = len(mat)
    
    # Vectorized diagonal extraction with wrapping
    indices = np.arange(n)
    row_indices = indices % n
    col_indices = (diag_index + indices) % n
    
    result = mat[row_indices, col_indices]
    
    return result.tolist() if isinstance(result, np.ndarray) else result


def rotate(vec, rotation):
    """Rotates vector to the left by rotation.
    
    Returns the rotated vector (v_i, v_(i+1), ..., v_(i-1)) of a vector v, 
    where i is the rotation.
    
    Args:
        vec (list): Vector.
        rotation (int): Index.
    
    Returns:
        Rotated vector.
    """
    # Convert to numpy array if needed
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    
    n = len(vec)
    
    # Vectorized rotation using numpy's roll
    result = np.roll(vec, -rotation)
    
    return result.tolist() if isinstance(result, np.ndarray) else result


def conjugate_matrix(matrix):
    """Conjugates all entries of matrix.
    
    Returns the conjugated matrix.
    
    Args:
        matrix (2-D list): Matrix.
    
    Returns:
        Conjugated matrix.
    """
    # Convert to numpy array if needed
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Vectorized conjugation
    conj_matrix = np.conjugate(matrix)
    
    return conj_matrix.tolist() if isinstance(conj_matrix, np.ndarray) else conj_matrix


def transpose_matrix(matrix):
    """Transposes a matrix.
    
    Returns the transposed matrix.
    
    Args:
        matrix (2-D list): Matrix.
    
    Returns:
        Transposed matrix.
    """
    # Convert to numpy array if needed
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Vectorized transpose
    transpose = matrix.T
    
    return transpose.tolist() if isinstance(transpose, np.ndarray) else transpose
