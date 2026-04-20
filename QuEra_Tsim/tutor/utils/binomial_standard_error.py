import  numpy as  np 
def binomial_standard_error(n, N):
    p = n / N
    if 0 < p < 1:
        return np.sqrt(p * (1 - p) / N)
    return 1 / N
