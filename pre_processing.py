import numpy as np
global RNG

def __smallest_gap(seq):
    seq = np.sort(seq)
    seq = np.abs(seq[1:]-seq[:-1])
    return seq[seq>0].min()

def add_uniform_noise(inputs, strength = 1.):
    inputs = inputs.copy().astype(np.float64)
    gaps = []
    for i in range(inputs.shape[1]):
        gaps.append(__smallest_gap(inputs[:, i]))
    for i in range(inputs.shape[1]):
        gap = gaps[i]*strength
        inputs[:, i] += np.random.uniform(-gap/2., gap/2., size=(inputs.shape[0],))
    return inputs

def discretize_counts(inputs, less_than):
    num_c = less_than + 1
    outputs = np.zeros((inputs.shape[0], inputs.shape[1]*num_c))
    for c in range(less_than):
        outputs[:, c::num_c] = (inputs==c).astype(int)
    outputs[:, less_than::num_c] = (inputs>c).astype(int)
    return outputs
