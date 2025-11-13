"""A module to encrypt for the CKKS scheme."""

import math
import numpy as np
from util.ciphertext import Ciphertext
import util.matrix_operations
from util.polynomial import Polynomial
from util.random_sample import sample_triangle


class CKKSBootstrappingContext:
    """An object that stores information necessary for bootstrapping.
    
    Attributes:
        poly_degree: Polynomial degree of ring.
        old_modulus: Original modulus of initial ciphertext.
        num_taylor_iterations: Number of iterations to perform for Taylor series
            for exp.
        encoding_mat0: Matrix for slot to coeff.
        encoding_mat1: Matrix for slot to coeff.
        encoding_mat_transpose0: Matrix for coeff to slot.
        encoding_mat_transpose1: Matrix for coeff to slot.
        encoding_mat_conj_transpose0: Matrix for coeff to slot.
        encoding_mat_conj_transpose1: Matrix for coeff to slot.
    """
    
    def __init__(self, params):
        """Generates private/public key pair for CKKS scheme.
        
        Args:
            params (CKKSParameters): Parameters including polynomial degree,
                ciphertext modulus, etc.
        """
        self.poly_degree = params.poly_degree
        self.old_modulus = params.ciph_modulus
        self.num_taylor_iterations = params.num_taylor_iterations
        self.generate_encoding_matrices()
    
    def get_primitive_root(self, index):
        """Returns the ith out of the n roots of unity, where n is 2 * poly_degree.
        
        Args:
            index (int): Index i to specify.
        
        Returns:
            The ith out of nth root of unity.
        """
        angle = math.pi * index / self.poly_degree
        return complex(math.cos(angle), math.sin(angle))
    
    def generate_encoding_matrices(self):
        """Generates encoding matrices for coeff_to_slot and slot_to_coeff operations.
        """
        num_slots = self.poly_degree // 2
        
        # NumPy optimization: vectorized primitive root computation
        power = 1
        powers = np.zeros(num_slots, dtype=np.int64)
        for i in range(num_slots):
            powers[i] = power
            power = (power * 5) % (2 * self.poly_degree)
        
        # Vectorized angle computation
        angles = np.pi * powers / self.poly_degree
        primitive_roots = np.cos(angles) + 1j * np.sin(angles)
        
        # Compute matrices for slot to coeff transformation using NumPy
        # encoding_mat0[i][k] = primitive_roots[i]^k
        k_indices = np.arange(num_slots)
        i_mesh, k_mesh = np.meshgrid(np.arange(num_slots), k_indices, indexing='ij')
        
        # Use object array to avoid overflow with complex numbers
        self.encoding_mat0 = [[1] * num_slots for _ in range(num_slots)]
        self.encoding_mat1 = [[1] * num_slots for _ in range(num_slots)]
        
        # Compute encoding_mat0: encoding_mat0[i][k] = primitive_roots[i]^k
        for i in range(num_slots):
            root_power = 1
            for k in range(num_slots):
                self.encoding_mat0[i][k] = root_power
                if k < num_slots - 1:
                    root_power = root_power * primitive_roots[i]
        
        # Compute encoding_mat1: starts from last element of encoding_mat0
        for i in range(num_slots):
            self.encoding_mat1[i][0] = self.encoding_mat0[i][-1] * primitive_roots[i]
        
        for i in range(num_slots):
            for k in range(1, num_slots):
                self.encoding_mat1[i][k] = self.encoding_mat1[i][k - 1] * primitive_roots[i]
        
        # Compute matrices for coeff to slot transformation
        self.encoding_mat_transpose0 = util.matrix_operations.transpose_matrix(self.encoding_mat0)
        self.encoding_mat_conj_transpose0 = util.matrix_operations.conjugate_matrix(
            self.encoding_mat_transpose0)
        
        self.encoding_mat_transpose1 = util.matrix_operations.transpose_matrix(self.encoding_mat1)
        self.encoding_mat_conj_transpose1 = util.matrix_operations.conjugate_matrix(
            self.encoding_mat_transpose1)
