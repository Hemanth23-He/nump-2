"""A module to encode integers as specified in the CKKS scheme - NumPy optimized version.
"""
import numpy as np
from util.ntt import FFTContext
from util.plaintext import Plaintext
from util.polynomial import Polynomial


class CKKSEncoder:
    """An encoder for several complex numbers as specified in the CKKS scheme.
    
    Attributes:
        degree (int): Degree of polynomial that determines quotient ring.
        fft (FFTContext): FFTContext object to encode/decode.
    """
    
    def __init__(self, params):
        """Inits CKKSEncoder with the given parameters.
        
        Args:
            params (Parameters): Parameters including polynomial degree,
                plaintext modulus, and ciphertext modulus.
        """
        self.degree = params.poly_degree
        self.fft = FFTContext(self.degree * 2)

    def encode(self, values, scaling_factor):
        """Encodes complex numbers into a polynomial.
        
        Encodes an array of complex number into a polynomial.
        
        Args:
            values (list): List of complex numbers to encode.
            scaling_factor (float): Scaling factor to multiply by.
        
        Returns:
            A Plaintext object which represents the encoded value.
        """
        # Convert to numpy array if needed
        if not isinstance(values, np.ndarray):
            values = np.array(values, dtype=np.complex128)
        
        num_values = len(values)
        plain_len = num_values << 1
        
        # Canonical embedding inverse variant
        to_scale = self.fft.embedding_inv(values)
        
        # Convert to numpy array if needed
        if not isinstance(to_scale, np.ndarray):
            to_scale = np.array(to_scale, dtype=np.complex128)
        
        # Vectorized: Multiply by scaling factor, and split up real and imaginary parts
        # Instead of loop:
        # for i in range(num_values):
        #     message[i] = int(to_scale[i].real * scaling_factor + 0.5)
        #     message[i + num_values] = int(to_scale[i].imag * scaling_factor + 0.5)
        
        # Vectorized version:
        real_part = np.round(to_scale.real * scaling_factor).astype(int)
        imag_part = np.round(to_scale.imag * scaling_factor).astype(int)
        
        # Concatenate real and imaginary parts
        message = np.concatenate([real_part, imag_part])
        
        return Plaintext(Polynomial(plain_len, message), scaling_factor)

    def decode(self, plain):
        """Decodes a plaintext polynomial.
        
        Decodes a plaintext polynomial back to a list of integers.
        
        Args:
            plain (Plaintext): Plaintext to decode.
        
        Returns:
            A decoded list of integers.
        """
        if not isinstance(plain, Plaintext):
            raise ValueError("Input to decode must be a Plaintext")
        
        # Get coefficients as numpy array
        coeffs = plain.poly.coeffs
        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(coeffs)
        
        plain_len = len(coeffs)
        num_values = plain_len >> 1
        
        # Vectorized: Divide by scaling factor, and turn back into a complex number
        # Instead of loop:
        # for i in range(num_values):
        #     message[i] = complex(plain.poly.coeffs[i] / plain.scaling_factor,
        #                          plain.poly.coeffs[i + num_values] / plain.scaling_factor)
        
        # Vectorized version:
        real_part = coeffs[:num_values] / plain.scaling_factor
        imag_part = coeffs[num_values:] / plain.scaling_factor
        
        # Create complex array
        message = real_part + 1j * imag_part
        
        # Compute canonical embedding variant
        return self.fft.embedding(message)
