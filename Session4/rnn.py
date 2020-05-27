import tensorflow.compat.v1 as tf
import numpy as np
from utils import gen_data_and_vocab, encode_data, MAX_DOC_LENGTH
import random

NUM_CLASSES = 20
class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        self._data = []
        self._labels = []
        self._sentence_lengths = []
        self._final_tokens = []
        for data_id, line in enumerate(d_lines):
            vector = []
            features = line.split('<fff>')
            label, doc_id, sentence_len = int(features[0]), int(features[1]), int(features[2])
            tokens = features[3].split()
            for token in tokens:
                vector.append(tf.int32(token))
            self._final_tokens.append(vector[-1])
            self._data.append(vector)
            self._labels.append(label)
        self._data = np.array(self._data)
        self._labels = np.array(self._labels)

        self._num_epoch = 0
        self._batch_id = 0
    #shuffle data and make batch size
    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.seed(2020)
            random.shuffle(indices)
            self._data, self._labels,self._sentence_lengths,self._final_tokens = self._data[indices], self._labels[indices],\
                                                                                self._sentence_lengths[indices], self._final_tokens[indices]
        return [np.array(self._data[start:end]), np.array(self._labels[start:end]), np.array(self._sentence_lengths[start:end]),\
               np.array(self._final_tokens[start:end])]
class RNN:
    def __init__(self, vocab_size, embedding_size, lstm_size, batch_size):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._lstm_size = lstm_size
        self._batch_size = batch_size

        self._data = tf.placeholder(tf.int32, shape=[batch_size, MAX_DOC_LENGTH])
        self._labels = tf.placeholder(tf.int32, shape=[batch_size, ])
        self._sentence_length = tf.placeholder(tf.int32, shape=[batch_size, ])
        self._final_tokens = tf.placeholder(tf.int32, shape=[batch_size, ])
    def embedding_layer(self, data):
        pretrained_vectors = []
        pretrained_vectors.append(np.zeros(self._embedding_size))
        np.random.seed(2020)
        for _ in range(self._vocab_size + 1):
            pretrained_vectors.append(np.random.normal(loc=0., scale=1., size=self._embedding_size))

        pretrained_vectors = np.array(pretrained_vectors)

        self._embedding_matrix = tf.get_variable(
            name='embedding',
            shape=(self._vocab_size+2, self._embedding_size),
            initializer=tf.constant_initializer(pretrained_vectors)
        )
        return tf.nn.embedding_lookup(self._embedding_matrix, data)
    def LSTM_layer(self, embeddings):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._lstm_size)
        zero_state = tf.zeros(shape=(self._batch_size, self._lstm_size))
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(zero_state, zero_state)

        lstm_input = tf.unstack(
            tf.transpose(embeddings, perm=[1,0,2])
        )
        lstm_output, last_state = tf.nn.static_rnn(
            cell=lstm_cell,
            inputs=lstm_input,
            initial_state=initial_state,
            sequence_length=self._sentence_length
        )
        lstm_output = tf.unstack(
            tf.transpose(lstm_output, perm=[1, 0 ,2])
        )
        lstm_output = tf.concat(
            lstm_output,
            axis=0
        )

        mask = tf.sequence_mask(
            lengths=self._sentence_length,
            maxlen=MAX_DOC_LENGTH,
            dtype=tf.float32
        ) #shape = (sentence_length
        mask = tf.concat(tf.unstack(mask, axis=0),axis=0)
        mask =tf.expand_dims(mask, axis=-1)

        lstm_output = lstm_output*mask
        lstm_output_split = tf.split(lstm_output, self._batch_size)
        lstm_output_sum = tf.reduce_sum(lstm_output_split, axis=1)
        lstm_output_average = lstm_output_sum / tf.expand_dims(
            tf.cast(self._sentence_length, tf.float32),
            -1
        )
        return lstm_output_average


    def build_graph(self):
        embeddings = self.embedding_layer(self._data)
        lstm_output = self.LSTM_layer(embeddings)

        final_weights = tf.get_variable(
            name="final_weights_output",
            shape=(self._lstm_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2020)
        )

        final_biases = tf.get_variable(
            name='final_biases_output',
            shape= (NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2020)
        )

        logits = tf.matmul(lstm_output, final_weights) + final_biases

        labels_one_hot = tf.one_hot(self._labels, NUM_CLASSES, dtype=tf.float32)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits
        )
        loss = tf.reduce_mean(loss)
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis = 1)
        predicted_labels = tf.squeeze(predicted_labels)
        return loss, predicted_labels

    def trainer(self, loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return train_op


def train_and_evaluate_RNN():
    with open('./datasets/w2v/vocab_w2v.txt') as f:
        vocab_size = len(f.read().splitlines())

    tf.set_random_seed(2020)
    rnn = RNN(
        vocab_size=vocab_size,
        embedding_size=300,
        lstm_size=50,
        batch_size=50
    )
    loss, predicted_labels = rnn.build_graph()
    train_op = rnn.trainer()
    with tf.Session() as sess:
        train_data_reader = DataReader(
            data_path='./datasets/w2v/20news-train-encoded.txt',
            batch_size=50
        )
        test_data_reader = DataReader(
            data_path='./datasets/w2v/20news-test-encoded.txt',
            batch_size=50
        )
        step = 0
        MAX_STEP = 1000

        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            next_train_batch = train_data_reader.next_batch()
            train_data, train_labels, sentence_length, final_tokens = next_train_batch
            plabels_eval, loss_eval = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict={
                    rnn._data: train_data,
                    rnn._labels: train_labels,
                    rnn._sentence_length: sentence_length,
                    rnn._final_tokens: final_tokens
                }
            )
            step += 1
            if step % 20 == 0:
                print('Step: {}, loss: {}'.format(step,loss_eval))
            if train_data_reader._current_



