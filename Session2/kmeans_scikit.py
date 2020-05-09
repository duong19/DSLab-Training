import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC

#kmeans algorithms
def clustering_with_Kmeans():
    data, labels = load_data('./datasets/20news-bydate/tf-idf-full-processed.txt')
    X = csr_matrix(data)
    print('========')
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2020
    ).fit(X)
    print("Purity Kmeans: ", compute_purity(kmeans.labels_, labels))
#linear SVM algorithms
def classifying_with_linear_SVMs():
    train_X, train_y = load_data('./datasets/20news-bydate/tf-idf-train.txt')
    classifier = LinearSVC(
        C=10.0,
        tol=0.001,
        verbose=True
    )
    classifier.fit(train_X, train_y)

    test_X, test_y = load_data('./datasets/20news-bydate/tf-idf-test.txt')
    predicted_y = classifier.predict(test_X)
    acc = compute_acc(predicted_y, test_y)
    print('Accuracy Linear SVM: ', acc)
#compute accuracy with test data
def compute_acc(predicted_y, y):
    matches = np.equal(predicted_y, y)
    acc = np.sum(matches.astype(float)) / y.size
    return acc
#compute purity of kmeans algorithms
def compute_purity(predicted_labels, labels):
    majority_sum = 0
    for c in range(20):
        member_labels = [labels[k] for k in range(len(predicted_labels)) if predicted_labels[k] == c]
        max_count = max([member_labels.count(i) for i in range(20)])
        majority_sum += max_count
    purity = majority_sum*1.0/len(labels)
    return purity
#load data from files
def load_data(path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        # create vector tf_idf for document d
        _r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            _r_d[index] = tfidf
        return np.array(_r_d)
    data = []
    labels = []
    with open(path) as f:
        d_lines = f.read().splitlines()
    for line in d_lines:
        features = line.split("<fff>")
        label = int(features[0])
        r_d = sparse_to_dense(features[2], vocab_size=14212)
        data.append(r_d)
        labels.append(label)

    return np.array(data), np.array(labels)

if __name__ =='__main__':
    clustering_with_Kmeans()
    classifying_with_linear_SVMs()
