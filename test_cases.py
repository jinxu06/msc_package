import numpy as np

class TestCase(object):

    def __init__(self):
        pass

    def generate_inputs(self):
        pass

    def generate_outputs(self):
        pass

    def check(self):
        pass


"""

class TestCase:

    def __init__(self, distris, percents=None):
        self.distris = distris
        if percents is None:
            percents = [1. / len(self.distris) for i in range(len(self.distris))]
        self.percents = percents

    def draw_inputs(self, num, distri):
        ori_shape = distri.shape
        data = []
        for i in range(num):
            s = np.random.multinomial(1, distri.flatten()).reshape(ori_shape)
            i, j = np.argwhere(s==1)[0]
            data.append([i, j])
        return np.array(data)

    def generate_training_data(self, num):
        nums = []
        for i in self.percents:
            nums.append(int(round(num * i)))
        nums[-1] -= np.sum(nums)-num
        data = []
        for i, d in enumerate(self.distris):
            inputs = self.draw_inputs(nums[i], d)
            targets = np.ones((nums[i], 1)) * i
            data.append(np.concatenate([inputs, targets], axis=1))
        data = np.concatenate(data, axis=0)
        data = data.astype(int)
        np.random.shuffle(data)
        return data[:,:-1], data[:,-1]

    def estimate_distribution(self, inputs, targets):
        for i in range(max(targets)+1):
            freq = np.zeros(self.distris[0].shape)
            data = inputs[targets==i, :]
            for d in data:
                freq[d[0], d[1]] += 1
            freq = freq / freq.sum()
            print freq


d1 = np.array([[0.1, 0.4],[0.4, 0.1]])
d2 = np.array([[0.2, 0.3],[0.3, 0.2]])

tc = TestCase([d1, d2])
inputs, targets = tc.generate_training_data(100000)
tc.estimate_distribution(inputs, targets)
"""
