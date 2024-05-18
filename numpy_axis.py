import numpy as np


layer_outputs = np.array([[4.8, 1.21, 2.385],
 [8.9, -1.81, 0.2],
 [1.41, 1.051, 0.026]])

print('Sum without axis')
print(np.sum(layer_outputs))
print('This will be identical to the above since default is None:')
print(np.sum(layer_outputs, axis=None))