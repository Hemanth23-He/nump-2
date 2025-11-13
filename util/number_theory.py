"""A module with number theory functions necessary for other functions - NumPy optimized version.
"""
import random
import sympy
import numpy as np


def mod_exp(val, exp, modulus):
    """Computes an exponent in a modulus.
    
    Raises val to power exp in the modulus without overflowing.
    
    Args:
        val (int): Value we wish to raise the power of.
        exp (int): Exponent.
        modulus (int): Modulus where computation is performed.
    
    Returns:
        A value raised to a power in a modulus.
    """
    # Python's built-in pow is already optimized for modular exponentiation
    # But we can add array support for batch operations
    if isinstance(val, np.ndarray):
        # Vectorized modular exponentiation for arrays
        return np.array([pow(int(v), int(exp), int(modulus)) for v in val])
    else:
        return pow(int(val), int(exp), int(modulus))


def mod_inv(val, modulus):
    """Finds an inverse in a given prime modulus.
    
    Finds the inverse of val in the modulus.
    
    Args:
        val (int): Value to find the inverse of.
        modulus (int): Modulus where computation is performed.
            Note: MUST BE PRIME.
    
    Returns:
        The inverse of the given value in the modulus.
    """
    # Using Fermat's Little Theorem: a^(-1) = a^(p-2) mod p
    if isinstance(val, np.ndarray):
        # Vectorized modular inverse for arrays
        return np.array([mod_exp(v, modulus - 2, modulus) for v in val])
    else:
        return mod_exp(val, modulus - 2, modulus)


def find_generator(modulus):
    """Finds a generator in the given modulus.
    
    Finds a generator, or primitive root, in the given prime modulus.
    
    Args:
        modulus (int): Modulus to find the generator in. Note: MUST
            BE PRIME.
    
    Returns:
        A generator, or primitive root in the given modulus.
    """
    # sympy's primitive_root is already optimized
    return sympy.ntheory.primitive_root(modulus)


def root_of_unity(order, modulus):
    """Finds a root of unity in the given modulus.
    
    Finds a root of unity with the given order in the given prime modulus.
    
    Args:
        order (int): Order n of the root of unity (an nth root of unity).
        modulus (int): Modulus to find the root of unity in. Note: MUST BE
            PRIME
    
    Returns:
        A root of unity with the given order in the given modulus.
    """
    if ((modulus - 1) % order) != 0:
        raise ValueError('Must have order q | m - 1, where m is the modulus. \
The values m = ' + str(modulus) + ' and q = ' + str(order) + ' do not satisfy this.')
    
    generator = find_generator(modulus)
    
    if generator is None:
        raise ValueError('No primitive root of unity mod m = ' + str(modulus))
    
    result = mod_exp(generator, (modulus - 1)//order, modulus)
    
    if result == 1:
        return root_of_unity(order, modulus)
    
    return result


def is_prime(number, num_trials=200):
    """Determines whether a number is prime.
    
    Runs the Miller-Rabin probabilistic primality test many times on the given number.
    
    Args:
        number (int): Number to perform primality test on.
        num_trials (int): Number of times to perform the Miller-Rabin test.
    
    Returns:
        True if number is prime, False otherwise.
    """
    if number < 2:
        return False
    if number != 2 and number % 2 == 0:
        return False
    
    # Find largest odd factor of n-1.
    exp = number - 1
    while exp % 2 == 0:
        exp //= 2
    
    # Miller-Rabin test with multiple trials
    for _ in range(num_trials):
        rand_val = int(random.SystemRandom().randrange(1, number))
        new_exp = exp
        power = pow(rand_val, new_exp, number)
        
        while new_exp != number - 1 and power != 1 and power != number - 1:
            power = (power * power) % number
            new_exp *= 2
        
        if power != number - 1 and new_exp % 2 == 0:
            return False
    
    return True


# Additional NumPy-optimized batch operations for CKKS

def batch_mod_exp(vals, exp, modulus):
    """Batch modular exponentiation for arrays.
    
    Optimized for processing multiple values at once.
    
    Args:
        vals (numpy.ndarray or list): Array of values to exponentiate.
        exp (int): Exponent.
        modulus (int): Modulus where computation is performed.
    
    Returns:
        numpy.ndarray of results.
    """
    if not isinstance(vals, np.ndarray):
        vals = np.array(vals)
    
    # Vectorized computation
    result = np.zeros_like(vals, dtype=np.int64)
    for i, val in enumerate(vals):
        result[i] = pow(int(val), int(exp), int(modulus))
    
    return result


def batch_mod_inv(vals, modulus):
    """Batch modular inverse for arrays.
    
    Optimized for processing multiple values at once.
    
    Args:
        vals (numpy.ndarray or list): Array of values to invert.
        modulus (int): Modulus where computation is performed.
    
    Returns:
        numpy.ndarray of results.
    """
    if not isinstance(vals, np.ndarray):
        vals = np.array(vals)
    
    # Vectorized computation using Fermat's Little Theorem
    return batch_mod_exp(vals, modulus - 2, modulus)
