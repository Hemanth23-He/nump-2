"""A module for CKKS bootstrapping pre-computations."""

import math
import numpy as np
from util.plaintext import Plaintext
from util.polynomial import Polynomial


class CKKSBootstrappingContext:
    """Pre-computed context for CKKS bootstrapping.
    
    This class pre-computes and stores the encoding matrices needed for
    the coefficient-to-slot and slot-to-coefficient transformations used
    in CKKS bootstrapping.
    
    Attributes:
        degree (int): Polynomial degree of ring.
        num_taylor_iterations (int): Number of iterations for Taylor approximation.
        encoding_mat0 (2-D Array): First encoding matrix.
        encoding_mat1 (2-D Array): Second encoding matrix.
        encoding_mat_transpose0 (2-D Array): Transpose of first encoding matrix.
        encoding_mat_transpose1 (2-D Array): Transpose of second encoding matrix.
        encoding_mat_conj_transpose0 (2-D Array): Conjugate transpose of first encoding matrix.
        encoding_mat_conj_transpose1 (2-D Array): Conjugate transpose of second encoding matrix.
    """
    
    def __init__(self, params):
        """Inits CKKSBootstrappingContext.
        
        Computes encoding matrices for bootstrapping transformations.
        
        Args:
            params (Parameters): CKKS parameters including polynomial degree.
        """
        self.degree = params.poly_degree
        self.num_taylor_iterations = self._compute_taylor_iterations()
        
        # Pre-compute encoding matrices using NumPy for efficiency
        self.encoding_mat0 = self._compute_encoding_matrix(0)
        self.encoding_mat1 = self._compute_encoding_matrix(1)
        
        # Compute transposes and conjugate transposes
        self.encoding_mat_transpose0 = self._transpose_matrix(self.encoding_mat0)
        self.encoding_mat_transpose1 = self._transpose_matrix(self.encoding_mat1)
        
        self.encoding_mat_conj_transpose0 = self._conjugate_transpose_matrix(self.encoding_mat0)
        self.encoding_mat_conj_transpose1 = self._conjugate_transpose_matrix(self.encoding_mat1)
    
    def _compute_taylor_iterations(self):
        """Computes optimal number of Taylor iterations based on degree.
        
        The number of iterations is chosen to balance accuracy and computational cost.
        
        Returns:
            int: Number of Taylor iterations for exponential approximation.
        """
        # Heuristic: log2(degree) - 3 iterations
        # For N=4096: log2(4096) - 3 = 12 - 3 = 9 iterations
        # For N=8192: log2(8192) - 3 = 13 - 3 = 10 iterations
        if self.degree <= 1024:
            return 6
        elif self.degree <= 2048:
            return 7
        elif self.degree <= 4096:
            return 8
        elif self.degree <= 8192:
            return 9
        elif self.degree <= 16384:
            return 10
        else:
            return 11
    
    def _compute_encoding_matrix(self, offset):
        """Computes encoding matrix for coefficient-to-slot transformation.
        
        This matrix implements the DFT-like transformation that converts
        polynomial coefficients into plaintext slots.
        
        Args:
            offset (int): Matrix offset (0 or 1) for splitting coefficients.
        
        Returns:
            2-D numpy array: Encoding matrix of size (degree//2) × (degree//2).
        """
        n = self.degree
        m = n // 2
        
        # NumPy optimization: vectorized computation
        # Create indices using meshgrid
        i_indices = np.arange(m, dtype=object)  # Use object for large values
        j_indices = np.arange(m, dtype=object)
        
        # Compute omega = e^(2πi/n), the primitive n-th root of unity
        omega = np.exp(2j * np.pi / n)
        
        # Compute encoding matrix elements: omega^((2*j + 1 + offset) * i)
        # Broadcasting creates m×m matrix efficiently
        i_mesh, j_mesh = np.meshgrid(i_indices, j_indices, indexing='ij')
        exponents = (2 * j_mesh + 1 + offset) * i_mesh
        
        # Compute matrix using vectorized operations
        encoding_matrix = np.power(omega, exponents)
        
        # Normalize by 1/sqrt(m) for proper DFT scaling
        encoding_matrix = encoding_matrix / np.sqrt(m)
        
        # Convert to list for compatibility with CKKS encoder
        return encoding_matrix.tolist()
    
    def _transpose_matrix(self, matrix):
        """Computes transpose of a matrix using NumPy.
        
        Args:
            matrix (2-D Array): Input matrix.
        
        Returns:
            2-D list: Transposed matrix.
        """
        # NumPy optimization: vectorized transpose
        np_matrix = np.array(matrix, dtype=complex)
        transposed = np_matrix.T
        return transposed.tolist()
    
    def _conjugate_transpose_matrix(self, matrix):
        """Computes conjugate transpose (Hermitian transpose) of a matrix.
        
        Args:
            matrix (2-D Array): Input matrix.
        
        Returns:
            2-D list: Conjugate transposed matrix.
        """
        # NumPy optimization: vectorized conjugate transpose
        np_matrix = np.array(matrix, dtype=complex)
        conj_transposed = np_matrix.conj().T
        return conj_transposed.tolist()
    
    def get_encoding_matrix_dim(self):
        """Returns the dimension of encoding matrices.
        
        Returns:
            int: Dimension (degree // 2).
        """
        return self.degree // 2
    
    def get_taylor_degree(self):
        """Returns the degree of Taylor polynomial used in bootstrapping.
        
        Returns:
            int: Taylor polynomial degree (typically 7 for CKKS).
        """
        # CKKS typically uses degree-7 Taylor approximation for sine
        return 7
    
    def get_taylor_coefficients(self):
        """Returns Taylor coefficients for sine approximation.
        
        The bootstrapping circuit uses Taylor approximation of:
        sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!
        
        Returns:
            list: Taylor coefficients [1, 0, -1/6, 0, 1/120, 0, -1/5040, 0].
        """
        # NumPy optimization: precompute factorials using vectorized operations
        coeffs = np.zeros(8, dtype=object)  # Object dtype for exact fractions
        
        # Sine Taylor series: sin(x) = Σ(-1)^n * x^(2n+1) / (2n+1)!
        coeffs[0] = 1          # x^1 / 1!
        coeffs[2] = -1/6       # -x^3 / 3!
        coeffs[4] = 1/120      # x^5 / 5!
        coeffs[6] = -1/5040    # -x^7 / 7!
        
        return coeffs.tolist()
    
    def compute_rotations_needed(self):
        """Computes list of rotation indices needed for bootstrapping.
        
        Returns:
            list: Sorted list of rotation amounts needed.
        """
        n = self.degree
        m = n // 2
        
        # Baby-Step Giant-Step algorithm requires specific rotations
        matrix_len = m
        matrix_len_factor1 = int(np.sqrt(matrix_len))
        if matrix_len != matrix_len_factor1 * matrix_len_factor1:
            matrix_len_factor1 = int(np.sqrt(2 * matrix_len))
        matrix_len_factor2 = matrix_len // matrix_len_factor1
        
        # NumPy optimization: vectorized rotation index generation
        rotations = set()
        
        # Small rotations (baby steps)
        rotations.update(np.arange(1, matrix_len_factor1).tolist())
        
        # Large rotations (giant steps)
        shifts = np.arange(matrix_len_factor2) * matrix_len_factor1
        rotations.update(shifts[shifts > 0].tolist())
        
        # Additional rotations for diagonal extraction
        for j in range(matrix_len_factor2):
            shift = matrix_len_factor1 * j
            for i in range(matrix_len_factor1):
                rot = shift + i
                if rot > 0:
                    rotations.add(rot)
        
        return sorted(list(rotations))
    
    def estimate_bootstrapping_depth(self):
        """Estimates multiplicative depth consumed by bootstrapping.
        
        Returns:
            int: Estimated depth (number of levels consumed).
        """
        # Depth breakdown:
        # - Coeff to slot: ~log(N) multiplications
        # - Taylor exp: ~log(iterations) + 7 multiplications
        # - Slot to coeff: ~log(N) multiplications
        
        depth_coeff_to_slot = int(np.log2(self.degree))
        depth_exp = self.num_taylor_iterations + 7  # Taylor polynomial degree
        depth_slot_to_coeff = int(np.log2(self.degree))
        
        total_depth = depth_coeff_to_slot + depth_exp + depth_slot_to_coeff
        return total_depth
    
    def estimate_bootstrapping_noise_growth(self):
        """Estimates noise growth during bootstrapping.
        
        Returns:
            float: Estimated noise growth factor (log scale).
        """
        # Noise grows exponentially with depth
        depth = self.estimate_bootstrapping_depth()
        
        # Heuristic: each multiplication adds ~2x noise, each rotation adds ~1.5x
        num_rotations = len(self.compute_rotations_needed())
        
        noise_from_mult = depth * np.log2(2)
        noise_from_rot = num_rotations * np.log2(1.5)
        
        total_noise_bits = noise_from_mult + noise_from_rot
        return float(total_noise_bits)
    
    def get_required_modulus_bits(self, security_bits=128):
        """Computes required ciphertext modulus for bootstrapping.
        
        Args:
            security_bits (int): Target security level (default 128).
        
        Returns:
            int: Required modulus size in bits.
        """
        # Modulus must support: security + noise growth + scaling factors
        depth = self.estimate_bootstrapping_depth()
        noise_bits = self.estimate_bootstrapping_noise_growth()
        
        # Scaling factor bits (typically 40-60 per level)
        scaling_bits_per_level = 50
        total_scaling_bits = depth * scaling_bits_per_level
        
        required_bits = security_bits + noise_bits + total_scaling_bits
        
        # Round up to next power of 2
        return int(2 ** np.ceil(np.log2(required_bits)))


def compute_encoding_matrix_fast(degree, offset=0):
    """Standalone function to compute encoding matrix quickly.
    
    This is a utility function for testing and verification.
    
    Args:
        degree (int): Polynomial degree.
        offset (int): Matrix offset (0 or 1).
    
    Returns:
        numpy array: Encoding matrix.
    """
    n = degree
    m = n // 2
    
    # Vectorized computation
    omega = np.exp(2j * np.pi / n)
    i_mesh, j_mesh = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
    exponents = (2 * j_mesh + 1 + offset) * i_mesh
    encoding_matrix = np.power(omega, exponents) / np.sqrt(m)
    
    return encoding_matrix


def verify_encoding_matrices(degree):
    """Verifies that encoding matrices satisfy DFT properties.
    
    Args:
        degree (int): Polynomial degree.
    
    Returns:
        bool: True if matrices are valid, False otherwise.
    """
    from ckks.ckks_parameters import CKKSParameters
    
    params = CKKSParameters(poly_degree=degree, 
                           ciph_modulus=1<<100, 
                           big_modulus=1<<200,
                           scaling_factor=1<<40)
    
    context = CKKSBootstrappingContext(params)
    
    # Check matrix dimensions
    m = degree // 2
    assert len(context.encoding_mat0) == m
    assert len(context.encoding_mat0[0]) == m
    
    # Check orthogonality (E * E^H ≈ I)
    E0 = np.array(context.encoding_mat0, dtype=complex)
    E0_H = np.array(context.encoding_mat_conj_transpose0, dtype=complex)
    
    product = E0 @ E0_H
    identity = np.eye(m)
    
    # Check if close to identity (within numerical precision)
    max_error = np.max(np.abs(product - identity))
    
    print(f"Matrix dimension: {m}×{m}")
    print(f"Orthogonality error: {max_error:.2e}")
    print(f"Taylor iterations: {context.num_taylor_iterations}")
    print(f"Bootstrapping depth: {context.estimate_bootstrapping_depth()}")
    print(f"Noise growth: {context.estimate_bootstrapping_noise_growth():.1f} bits")
    
    return max_error < 1e-10


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("CKKS Bootstrapping Context - NumPy Optimized")
    print("=" * 60)
    
    # Test with different polynomial degrees
    for N in [2048, 4096, 8192]:
        print(f"\n{'='*60}")
        print(f"Testing with N = {N}")
        print(f"{'='*60}")
        
        verify_encoding_matrices(N)
        
        # Create context
        from ckks.ckks_parameters import CKKSParameters
        params = CKKSParameters(poly_degree=N, 
                               ciph_modulus=1<<300, 
                               big_modulus=1<<600,
                               scaling_factor=1<<40)
        
        context = CKKSBootstrappingContext(params)
        
        # Display bootstrapping requirements
        print(f"\nRotations needed: {len(context.compute_rotations_needed())}")
        print(f"Required modulus: {context.get_required_modulus_bits()} bits")
        print(f"Memory per matrix: {N * N * 16 / (1024**2):.2f} MB")
