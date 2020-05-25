from collections import defaultdict

import numpy as np



class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d                             #vector tf_idf of document d
        self._label = label                         #news group id of document d
        self._doc_id = doc_id                       #file id of document d

class Cluster:
    def __init__(self):
        self._centroid = None                       #centroid of cluster
        self._members = []                          #list of members of cluster
    def reset_members(self):                        #reset list of member to empty
        self._members = []
    def add_members(self, member):                  #add member to list of member
        self._members.append(member)

class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters                                       #number of clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]         #list of clusters
        self._E = []                                                            #list of centroids
        self._S = 0                                                             #similarity of all clusters
    # load data from file
    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
        #create vector tf_idf for document d
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(':')[0])
                tfidf = float(index_tfidf.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open('./datasets/20news-bydate/words_idfs.txt','r') as f:
            vocab_size = len(f.read().splitlines())
        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))
    # random initialize value for clusters from dataset
    def random_init(self, seed_value):
        np.random.seed(seed_value)
        data = np.array(self._data)
        data_centroids = data[np.random.choice(range(len(self._data)), self._num_clusters, replace=False)]
        list(data_centroids)
        for i, member in enumerate(data_centroids):
            self._clusters[i]._centroid = member._r_d
    def compute_similarity(self,member, centroid):
        s = np.dot(member._r_d, centroid)
        return s
    #run kmeans algorithms
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
                print('NMI: ', self.compute_NMI())
                print('Purity: ', self.compute_purity())
                break
    #assign member to cluster
    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                max_similarity = similarity
                best_fit_cluster = cluster
        best_fit_cluster.add_members(member)
        return max_similarity
    def update_centroid_of(self, cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        aver_r_d = np.mean(member_r_ds, axis = 0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d ** 2))
        new_centroid = np.array([value/sqrt_sum_sqr for value in aver_r_d])
        cluster._centroid = new_centroid
    #select stopping condition for kmeans
    def stopping_condition(self, criterion, threshold):
        criteria = ["centroid", "similarity", "max_iters"]
        assert criterion in criteria
        if criterion == "max_iters":
            if self._iteration >= threshold:
                return True
            else:
                return False
        elif criterion == "centroid":
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
            self._E = E_new
            if (len(E_new_minus_E) <= threshold):
                return True
            else:
                return False
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False
    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1. / len(self._data)

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.
            H_omega += - wk / N * np.log10(wk / N)
            member_labels = [member._label
                             for member in cluster._members]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2. / (H_omega + H_C)


if __name__ == "__main__":
    kmeans = Kmeans(20)
    kmeans.load_data("./datasets/20news-bydate/tf-idf-full-processed.txt")
    kmeans.run(2019, 'max_iters', 30)

        