import numpy as np

a  = np.ndarray((3,3))

pad_a = np.pad(a, ((1,1),(1,1)), 'constant', constant_values=0)

print(pad_a)
print(pad_a.shape)