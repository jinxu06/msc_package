import numpy as np
#import tensorflow as tf
#from classifiers import *

def build_discriminate_dataset(original_inputs, original_targets, generated_inputs, generated_targets, sampling_size=None, encode_targets=True, split=0.5):

    if encode_targets:
        m = int(max(original_targets))+1
        ori_targets = np.zeros((original_targets.shape[0], m), dtype=np.int32)
        gen_targets = np.zeros((generated_targets.shape[0], m), dtype=np.int32)
        for i in range(m):
            ori_targets[:, i] = original_targets==i
            gen_targets[:, i] = generated_targets==i
        ori_targets = ori_targets.astype(int)
        gen_targets = gen_targets.astype(int)
        original_targets = ori_targets
        generated_targets = gen_targets
    else:
        original_targets = original_targets[:,None]
        generated_targets = generated_targets[:,None]

    ori_data = np.concatenate([original_inputs, original_targets], axis=1)
    gen_data = np.concatenate([generated_inputs, generated_targets], axis=1)
    if sampling_size is not None:
        np.random.shuffle(ori_data)
        ori_data = ori_data[:sampling_size, :]
        np.random.shuffle(gen_data)
        gen_data = gen_data[:sampling_size, :]

    ori_data = np.concatenate([ori_data, np.ones((ori_data.shape[0], 1), dtype=np.int32)], axis=1)
    gen_data = np.concatenate([gen_data, np.zeros((gen_data.shape[0], 1), dtype=np.int32)], axis=1)

    data = np.concatenate([ori_data, gen_data], axis=0)
    np.random.shuffle(data)
    train_data = data[:int(data.shape[0]*split), :]
    valid_data = data[int(data.shape[0]*split):, :]
    return train_data[:, :-1], train_data[:, -1], valid_data[:, :-1], valid_data[:, -1]

"""
class SyntheticDataDiscriminator:

    def __init__(self, classifier, sampling_size=None):
        self.sampling_size = sampling_size
        self.classifier = classifier

    def feed_data(self, original_inputs, original_targets, generated_inputs, generated_targets, encode_targets=True):

        if encode_targets:
            m = int(max(original_targets))+1
            ori_targets = np.zeros((original_targets.shape[0], m), dtype=np.int32)
            gen_targets = np.zeros((generated_targets.shape[0], m), dtype=np.int32)
            for i in range(m):
                ori_targets[:, i] = original_targets==i
                gen_targets[:, i] = generated_targets==i
            ori_targets = ori_targets.astype(int)
            gen_targets = gen_targets.astype(int)
            original_targets = ori_targets
            generated_targets = gen_targets
        else:
            original_targets = original_targets[:,None]
            generated_targets = generated_targets[:,None]

        ori_data = np.concatenate([original_inputs, original_targets], axis=1)
        gen_data = np.concatenate([generated_inputs, generated_targets], axis=1)
        if self.sampling_size is not None:
            np.random.shuffle(ori_data)
            ori_data = ori_data[:self.sampling_size, :]
            np.random.shuffle(gen_data)
            gen_data = gen_data[:self.sampling_size, :]

        ori_data = np.concatenate([ori_data, np.ones((ori_data.shape[0], 1), dtype=np.int32)], axis=1)
        gen_data = np.concatenate([gen_data, np.zeros((gen_data.shape[0], 1), dtype=np.int32)], axis=1)

        data = np.concatenate([ori_data, gen_data], axis=0)
        np.random.shuffle(data)
        self.train_data = data[:data.shape[0]/2, :]
        self.valid_data = data[data.shape[0]/2:, :]
        return data

    def discriminate(self, verbose=1):
        return self.classifier.experiment(self.train_data[:, :-1], self.train_data[:, -1], self.valid_data[:, :-1], self.valid_data[:, -1], verbose=verbose)
"""
