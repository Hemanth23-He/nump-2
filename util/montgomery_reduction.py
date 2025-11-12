class MontgomeryReducer:
    def __init__(self, modulus):
        self.modulus = modulus
        self.r = 1 << (modulus.bit_length() + 1)  # R > modulus
        self.r_inv = pow(self.r, -1, modulus)
        self.np = pow(-modulus, -1, self.r)
    
    def to_montgomery(self, x):
        return (x * self.r) % self.modulus
    
    def from_montgomery(self, x):
        return (x * self.r_inv) % self.modulus

    def reduce(self, x):
        m = ((x & (self.r - 1)) * self.np) & (self.r - 1)
        t = (x + m * self.modulus) >> self.r.bit_length() - 1
        if t >= self.modulus:
            t -= self.modulus
        return t

    def inverse(self, x):
        # Modular inverse as before
        a, b, u = x, self.modulus, 1
        while b != 0:
            t = a // b
            a, b = b, a - t * b
            u = u - t * u
        return u % self.modulus
