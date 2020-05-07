import regex as re
import numpy as np
from sklearn.linear_model import Ridge


def extract_data(readfile,writefile):
    with open(readfile,'r') as fread:
        with open(writefile,'w') as fwrite:
            for line in fread:
                if (re.match("(?<=^| )\W+\d+(\.\d+)?(?=$| )", line) or re.match("(?<=^| )\d+\W+(\.\d+)?(?=$| )",line)) and re.match("^[^#]+$",line):
                    fwrite.write(line)

def read_data(filename):
    with open(filename, 'r') as f:
        X = f.readline()
        X = re.sub("\n", "", X)
        X = np.array([float(i) for i in X.split()])
        i = 0
        for line in f:
            line = re.sub("\n", "", line)
            line = np.array([float(i) for i in line.split()])
            X = np.concatenate((X,line),axis=0)
            i = i + 1
        X = X.reshape(i+1, 17)
        X_all = X[:, 1:-1]
        y_all = X[:,-1]
    return X_all, y_all

def normalized_and_add_one(X):
    X_max = np.array([[np.amax(X[:, id]) for id in range(X.shape[1])] for _ in range(X.shape[0])])
    X_min = np.array([[np.amin(X[:, id]) for id in range(X.shape[1])] for _ in range(X.shape[0])])

    X_normalized = (X - X_min)/(X_max - X_min)
    X_normalized = np.concatenate((np.ones((X.shape[0], 1)), X_normalized),axis = 1)
    return X_normalized

def normalized(y):
    y_max = np.max(y_all)
    y_min = np.min(y_all)
    y_normalized = (y - y_min)/(y_max - y_min)
    return y_normalized
class RidgeRegression:
    def __init__(self):
        return
    def fit(self, X_train, y_train, LAMBDA):
        # print(X_train.shape, y_train.shape)
        assert len(X_train.shape) == 2 and X_train.shape[0] == y_train.shape[0]
        W = np.linalg.inv(np.dot(X_train.T, X_train) + LAMBDA*np.identity(X_train.shape[1])).dot(X_train.T).dot(y_train)
        return W
    def fit_gradient_descent(self, X_train, y_train, LAMBDA, learning_rate=0.005, max_num_epoch = 100, batch_size = 5):
        W = np.random.randn(X_train.shape[1])
        last_loss = 10e+8
        for epoch in range(max_num_epoch):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            y_train = y_train[arr]
            total_minibatch = int(np.ceil(X_train.shape[0]/batch_size))
            for i in range(total_minibatch):
                index = i*batch_size
                X_train_sub = X_train[index:index + batch_size]
                y_train_sub = y_train[index:index + batch_size]
                grad = X_train_sub.T.dot(X_train_sub.dot(W) - y_train_sub) + LAMBDA * W
                W = W - learning_rate * grad
            new_loss = self.compute_RSS(self.predict(W, X_train), y_train)
            if (np.abs(new_loss - last_loss) <= 1e-5):
                break
            last_loss = new_loss
        return W
    def compute_RSS(self,y, y_hat):
        loss = (1. / y.shape[0]) * np.sum((y - y_hat)**2)
        return loss
    def predict(self, W, X):
        y_hat = X.dot(W)
        return y_hat
    def get_the_best_LAMBDA(self, X_train, y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
            # print(valid_ids, len(row_ids))
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            aver_RSS = 0
            # print(valid_ids)
            for i in range(num_folds):
                valid_part = {'X' : X_train[valid_ids[i]], 'Y' : y_train[valid_ids[i]]}
                train_part = {'X' : X_train[train_ids[i]], 'Y' : y_train[train_ids[i]]}
                # print(train_part['Y'], i)
                W = self.fit_gradient_descent(train_part['X'], train_part['Y'], LAMBDA)
                y_hat = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], y_hat)
            return aver_RSS/num_folds
        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS

        best_LAMBDA, minimum_RSS = range_scan(0, 10000 ** 2, range(50))
        LAMBDA_values = [k*1./1000 for k in range(max(0, (best_LAMBDA - 1)*1000), (best_LAMBDA + 1) * 1000,1)]
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values)
        return best_LAMBDA

if __name__ == '__main__':
    X_all, y_all = read_data("datasets/data.txt")
    X_all = normalized_and_add_one(X_all)
    #y_all = normalized(y_all)
    X_train, y_train = X_all[:50], y_all[:50]
    X_test, y_test = X_all[50:], y_all[50:]
    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, y_train)
    print('Best LAMBDA: ', best_LAMBDA)
    W_learned = ridge_regression.fit_gradient_descent(X_train=X_train, y_train=y_train, LAMBDA=best_LAMBDA)
    y_predict = ridge_regression.predict(W=W_learned, X=X_test)
    print(ridge_regression.compute_RSS(y_test, y_predict))


