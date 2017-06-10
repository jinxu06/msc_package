import numpy as np
import tensorflow as tf
from classifiers import *



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

    def discriminate(self, quiet=False):
        return self.classifier.experiment(self.train_data[:, :-1], self.train_data[:, -1], self.valid_data[:, :-1], self.valid_data[:, -1], quiet=quiet)

        """
        if self.method == 'SVM':
            print "SVM discriminator ------"
            from sklearn.svm import SVC
            best = 0.
            for c in [0.05, 0.1, 0.5, 1, 2, 4, 8, 16, 32]:
                clf = SVC(C=c)
                clf.fit(self.train_data[:, :-1], self.train_data[:, -1])
                pred = clf.predict(self.valid_data[:, :-1])
                cp = pred == self.valid_data[:, -1]
                acc = np.sum(cp) / float(len(cp))
                if acc > best:
                    best = acc
                print "C={0}, Acc={1}".format(c, acc)
            print "Best Accuracy:{0}".format(best)

        if self.method == 'RandomForest':
            print "RandomForest discriminator ------"
            from sklearn.ensemble import RandomForestClassifier
            best = 0.
            for n_estimators in [1, 5, 10, 50, 100]:
                rf = RandomForestClassifier(n_estimators=n_estimators)
                rf.fit(self.train_data[:, :-1], self.train_data[:, -1])
                pred = rf.predict(self.valid_data[:, :-1])
                cp = pred == self.valid_data[:, -1]
                acc = np.sum(cp) / float(len(cp))
                if acc > best:
                    best = acc
                print "n_estimators={0}, Acc={1}".format(n_estimators, acc)
            print "Best Accuracy:{0}".format(best)
        """
