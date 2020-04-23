from collections import defaultdict

import numpy as np


class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id

class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []
    def reset_members(self):
        self._members = []
    def add_members(self, member):
        self._members.append(member)

class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        self._E = []
        self._S = []
    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(':')[0])
                tfidf = float(index_tfidf.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open('./datasets/20news-bydate/word-idfs.txt') as f:
            vocab_size = len(f.read().splitlines())
        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))
    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)

        self._iteration = 0
        while True:
            for cluster in self._clusters:
                cluster.reset_members()
                self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            self._iteration += 1
            if self.stopping_condition(criterion, threshold):
                break
    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity
                max_similarity = similarity
                best_fit_cluster = cluster
        best_fit_cluster.add_members(member)
        return  max_similarity
    def update_centroid_of(self, cluster):
        