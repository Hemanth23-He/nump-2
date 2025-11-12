"""A module to handle polynomial arithmetic in the quotient ring
Z_a[x]/f(x) - NumPy optimized version.
"""
import numpy as np
from util.ntt import NTTContext, FFTContext


class Polynomial:
    """A polynomial in the ring R_a.
    
    Here, R is the quotient ring Z[x]/f(x), where f(x) = x^d + 1.
    The polynomial keeps track of the ring degree d, the coefficient
    modulus a, and the coefficients in an array.
    
    Attributes:
        ring_degree (int): Degree d of polynomial that determines the
            quotient ring R.
        coeffs (numpy.ndarray): Array of coefficients of polynomial, where coeffs[i]
            is the coefficient for x^i.
    """
    
    def __init__(self, degree, coeffs):
        """Inits Polynomial in the ring R_a with the given coefficients.
        
        Args:
            degree (int): Degree of quotient polynomial for ring R_a.
            coeffs (array-like): Array of integers of size degree, representing
                coefficients of polynomial.
        """
        self.ring_degree = degree
        
        # Convert to numpy array if not already
        if isinstance(coeffs, np.ndarray):
            self.coeffs = coeffs.copy()
        else:
            self.coeffs = np.array(coeffs, dtype=np.float64)
            
        assert len(self.coeffs) == degree, 'Size of polynomial array %d is not \
equal to degree %d of ring' % (len(self.coeffs), degree)

    def add(self, poly, coeff_modulus=None):
        """Adds two polynomials in the ring.
        
        Adds the current polynomial to poly inside the ring R_a.
        
        Args:
            poly (Polynomial): Polynomial to be added to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.
        
        Returns:
            A Polynomial which is the sum of the two polynomials.
        """
        assert isinstance(poly, Polynomial)
        
        # Vectorized addition
        new_coeffs = self.coeffs + poly.coeffs
        
        if coeff_modulus:
            new_coeffs = np.mod(new_coeffs, coeff_modulus)
            
        return Polynomial(self.ring_degree, new_coeffs)

    def subtract(self, poly, coeff_modulus=None):
        """Subtracts second polynomial from first polynomial in the ring.
        
        Computes self - poly.
        
        Args:
            poly (Polynomial): Polynomial to be added to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.
        
        Returns:
            A Polynomial which is the difference between the two polynomials.
        """
        assert isinstance(poly, Polynomial)
        
        # Vectorized subtraction
        new_coeffs = self.coeffs - poly.coeffs
        
        if coeff_modulus:
            new_coeffs = np.mod(new_coeffs, coeff_modulus)
            
        return Polynomial(self.ring_degree, new_coeffs)

    def multiply(self, poly, coeff_modulus, ntt=None, crt=None):
        """Multiplies two polynomials in the ring using NTT.
        
        Multiplies the current polynomial to poly inside the ring R_a
        using the Number Theoretic Transform (NTT) in O(nlogn).
        
        Args:
            poly (Polynomial): Polynomial to be multiplied to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.
            ntt (NTTContext): An instance of the NTTContext object, which
                can be used for multiplication.
            crt (CRTContext): An instance of the CRTContext object, which
                was created with primes whose product is the coefficient
                modulus. It defaults to None, if we are not using the
                CRT representation.
        
        Returns:
            A Polynomial which is the product of the two polynomials.
        """
        if crt:
            return self.multiply_crt(poly, crt)
            
        if ntt:
            a = ntt.ftt_fwd(self.coeffs)
            b = ntt.ftt_fwd(poly.coeffs)
            # Vectorized element-wise multiplication
            ab = a * b
            prod = ntt.ftt_inv(ab)
            return Polynomial(self.ring_degree, prod)
        
        return self.multiply_naive(poly, coeff_modulus)

    def multiply_crt(self, poly, crt):
        """Multiplies two polynomials in the ring in CRT representation.
        
        Multiplies the current polynomial to poly inside the ring by
        splitting it into Chinese Remainder Theorem subrings for the primes
        given. For each subring, we multiply using NTT and recombine with CRT.
        
        Args:
            poly (Polynomial): Polynomial to be multiplied to the current
                polynomial.
            crt (CRTContext): An instance of the CRTContext object, which
                was created with primes whose product is the coefficient
                modulus.
        
        Returns:
            A Polynomial which is the product of the two polynomials.
        """
        assert isinstance(poly, Polynomial)
        
        poly_prods = []
        
        # Perform NTT for each prime factor.
        for i in range(len(crt.primes)):
            prod = self.multiply(poly, crt.primes[i], ntt=crt.ntts[i])
            poly_prods.append(prod)
        
        # Combine the products with CRT - vectorized
        # Stack all coefficients into a 2D array for batch CRT reconstruction
        coeff_matrix = np.array([p.coeffs for p in poly_prods])
        
        final_coeffs = np.zeros(self.ring_degree)
        for i in range(self.ring_degree):
            values = coeff_matrix[:, i]
            final_coeffs[i] = crt.reconstruct(values)
        
        return Polynomial(self.ring_degree, final_coeffs).mod_small(crt.modulus)

    def multiply_fft(self, poly, round=True):
        """Multiplies two polynomials in the ring using FFT.
        
        Multiplies the current polynomial to poly inside the ring R_a
        using FFT.
        
        Args:
            poly (Polynomial): Polynomial to be multiplied to the current
                polynomial.
        
        Returns:
            A Polynomial which is the product of the two polynomials.
        """
        assert isinstance(poly, Polynomial)
        
        fft = FFTContext(self.ring_degree * 8)
        
        # Pad with zeros
        a_padded = np.concatenate([self.coeffs, np.zeros(self.ring_degree)])
        b_padded = np.concatenate([poly.coeffs, np.zeros(self.ring_degree)])
        
        a = fft.fft_fwd(a_padded)
        b = fft.fft_fwd(b_padded)
        
        # Vectorized element-wise multiplication
        ab = a * b
        prod = fft.fft_inv(ab)
        
        # Reduction modulo x^d + 1 - vectorized
        poly_prod = np.zeros(self.ring_degree)
        
        for d in range(2 * self.ring_degree - 1):
            # Since x^d = -1, the degree is taken mod d, and the sign
            # changes when the exponent is > d.
            index = d % self.ring_degree
            sign = (int(d < self.ring_degree) - 0.5) * 2
            poly_prod[index] += sign * prod[d]
        
        if round:
            return Polynomial(self.ring_degree, poly_prod).round()
        else:
            return Polynomial(self.ring_degree, poly_prod)

    def multiply_naive(self, poly, coeff_modulus=None):
        """Multiplies two polynomials in the ring in O(n^2).
        
        Multiplies the current polynomial to poly inside the ring R_a
        naively in O(n^2) time.
        
        Args:
            poly (Polynomial): Polynomial to be multiplied to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.
        
        Returns:
            A Polynomial which is the product of the two polynomials.
        """
        assert isinstance(poly, Polynomial)
        
        poly_prod = np.zeros(self.ring_degree)
        
        for d in range(2 * self.ring_degree - 1):
            # Since x^d = -1, the degree is taken mod d, and the sign
            # changes when the exponent is > d.
            index = d % self.ring_degree
            sign = int(d < self.ring_degree) * 2 - 1
            
            # Perform a convolution to compute the coefficient for x^d.
            # Vectorized using numpy's dot product
            valid_indices = np.arange(max(0, d - self.ring_degree + 1), 
                                     min(d + 1, self.ring_degree))
            
            if len(valid_indices) > 0:
                coeff = np.dot(self.coeffs[valid_indices], 
                             poly.coeffs[d - valid_indices])
                poly_prod[index] += sign * coeff
            
            if coeff_modulus:
                poly_prod[index] %= coeff_modulus
        
        return Polynomial(self.ring_degree, poly_prod)

    def scalar_multiply(self, scalar, coeff_modulus=None):
        """Multiplies polynomial by a scalar.
        
        Multiplies the current polynomial to scalar inside the ring R_a.
        
        Args:
            scalar (int/float): Scalar to be multiplied to the current
                polynomial.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.
        
        Returns:
            A Polynomial which is the product of the polynomial and the
            scalar.
        """
        # Vectorized scalar multiplication
        new_coeffs = scalar * self.coeffs
        
        if coeff_modulus:
            new_coeffs = np.mod(new_coeffs, coeff_modulus)
        
        return Polynomial(self.ring_degree, new_coeffs)

    def scalar_integer_divide(self, scalar, coeff_modulus=None):
        """Divides polynomial by a scalar.
        
        Performs integer division on the current polynomial by the scalar inside
        the ring R_a.
        
        Args:
            scalar (int): Scalar to be divided by.
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.
        
        Returns:
            A Polynomial which is the quotient of the polynomial and the
            scalar.
        """
        # Vectorized integer division
        new_coeffs = self.coeffs // scalar
        
        if coeff_modulus:
            new_coeffs = np.mod(new_coeffs, coeff_modulus)
        
        return Polynomial(self.ring_degree, new_coeffs)

    def rotate(self, r):
        """Rotates plaintext coefficients by r.
        
        Rotates all the plaintext coefficients to the left such that the x^r
        coefficient is now the coefficient for x^0. We do so by applying the
        transformation m(X) -> m(X^k), where k = 5^r in the ciphertext
        polynomial.
        
        Returns:
            A rotated Polynomial.
        """
        k = 5 ** r
        new_coeffs = np.zeros(self.ring_degree)
        
        # Vectorized index computation
        indices = (np.arange(self.ring_degree) * k) % (2 * self.ring_degree)
        
        # Split into two cases based on index value
        mask_low = indices < self.ring_degree
        mask_high = ~mask_low
        
        new_coeffs[indices[mask_low]] = self.coeffs[mask_low]
        new_coeffs[indices[mask_high] - self.ring_degree] = -self.coeffs[mask_high]
        
        return Polynomial(self.ring_degree, new_coeffs)

    def conjugate(self):
        """Conjugates plaintext coefficients.
        
        Conjugates all the plaintext coefficients. We do so by applying the
        transformation m(X) -> m(X^{-1}).
        
        Returns:
            A conjugated Polynomial.
        """
        new_coeffs = np.zeros(self.ring_degree)
        new_coeffs[0] = self.coeffs[0]
        
        # Vectorized conjugation
        new_coeffs[1:] = -self.coeffs[-1:0:-1]
        
        return Polynomial(self.ring_degree, new_coeffs)

    def round(self):
        """Rounds all coefficients to nearest integer.
        
        Rounds all the current polynomial's coefficients to the nearest
        integer, where |x| = n + 0.5 rounds to |x| = n
        (i.e. 0.5 rounds to 0 and -1.5 rounds to -1).
        
        Returns:
            A Polynomial which is the rounded version of the current
            polynomial.
        """
        if np.iscomplexobj(self.coeffs):
            new_coeffs = np.round(self.coeffs.real)
        else:
            new_coeffs = np.round(self.coeffs)
        
        return Polynomial(self.ring_degree, new_coeffs)

    def floor(self):
        """Rounds all coefficients down to nearest integer.
        
        Rounds all the current polynomial's coefficients down to the nearest
        integer.
        
        Returns:
            A Polynomial which is the floor of the current
            polynomial.
        """
        new_coeffs = np.floor(self.coeffs).astype(int)
        return Polynomial(self.ring_degree, new_coeffs)

    def mod(self, coeff_modulus):
        """Mods all coefficients in the given coefficient modulus.
        
        Mods all coefficients of the current polynomial using the
        given coefficient modulus.
        
        Args:
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.
        
        Returns:
            A Polynomial whose coefficients are modulo coeff_modulus.
        """
        # Vectorized modulo operation
        new_coeffs = np.mod(self.coeffs, coeff_modulus)
        return Polynomial(self.ring_degree, new_coeffs)

    def mod_small(self, coeff_modulus):
        """Turns all coefficients in the given coefficient modulus
        to the range (-q/2, q/2].
        
        Turns all coefficients of the current polynomial
        in the given coefficient modulus to the range (-q/2, q/2].
        
        Args:
            coeff_modulus (int): Modulus a of coefficients of polynomial
                ring R_a.
        
        Returns:
            A Polynomial whose coefficients are modulo coeff_modulus.
        """
        try:
            # Vectorized modulo and centering
            new_coeffs = np.mod(self.coeffs, coeff
