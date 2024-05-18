import math

import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exponentiated_values = []

for output in layer_outputs:
    exponentiated_values.append(math.e ** output)
    
print('exponentiated values:')
print(exponentiated_values)

# my attempt without looking up correct form
# normalized_values = []
# for exp in exponentiated_values:
#     normalized_values.append(exp / len(exponentiated_values))

# correct form of normalizing the exponentiated values
norm_base = sum(exponentiated_values)
norm_values = []
for val in exponentiated_values:
    norm_values.append(val / norm_base)

print("normalized values")
print(norm_values)
print(f"Sum of normalized values {sum(norm_values)}")


print('\n\nnumpy version\n')
layer_outputs = [4.8, 1.21, 2.385]
# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)
# Now normalize values
norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values)
print('sum of normalized values:', np.sum(norm_values))
