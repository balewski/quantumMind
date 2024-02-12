import numpy as np

# Number of classes
n = 3

# True and predicted probabilities for uniform distribution
p_true = np.full(n, 1 / n)
p_pred = np.full(n, 1 / n)

# Calculating cross-entropy
cross_entropy = -np.sum(p_true * np.log(p_pred))
print("N=%d Cross Entropy: %.2f"%(n,cross_entropy))
