from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy, categorical_crossentropy
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import random
import argparse
from sentiment_model import create_model

# randomly shuffle x, y lists
def shuffle_lists(X, y):

    shuff_list = list(zip(X, y))
    random.shuffle(shuff_list, )
    X, y = zip(*shuff_list)
    return X, y


# replace list of words with ids from word_index
def process_batch_sentences(batch, word_index):

    batch = list(map(lambda d: list(
        map(lambda w: word_index.get(w, word_index["unkid"]), d)), batch))
    return np.array(batch)


# segmentate data by sentence length
def segmentate_data(x_data, y_data):

    x_dict = dict()
    y_dict = dict()

    for x, y in zip(x_data, y_data):
        key = len(x)
        if key not in x_dict:
            x_dict[key] = list()
            y_dict[key] = list()

        x_dict[key].append(x)
        y_dict[key].append(y)

    return x_dict, y_dict

# get dataset, vocab data and embeddings
def get_data(vocab_size, step):

    # load embeddings
    with open('embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)

    # read word index
    with open('word_index.txt', 'r') as f:
        lines = f.readlines()

    word_index = dict()
    for line in lines[:vocab_size]:
        k, v = line.split()
        word_index[k] = int(v)

    if step == 'test':

        with open('final_test_datset.txt', 'r') as f:
            data = f.readlines()

        X = list()
        for dato in data:
            X.append(dato.strip().split())

        final_data = segmentate_data(X, list(range(len(X))))

    else:

        # read train dataset
        with open('final_train_dataset.txt', 'r') as f:
            data = f.readlines()

        X = list()
        y = list()

        for dato in data:
            dato = dato.strip().split()
            X.append(dato[:-1])
            value = [0, 0]
            value[int(dato[-1])] = 1
            y.append(value)

        X_train, X_val, y_train, y_val = \
            train_test_split(X, y, test_size=0.02, random_state=42)

        train_data = segmentate_data(X_train, y_train)
        val_data = segmentate_data(X_val, y_val)

        final_data = train_data, val_data

    return final_data, embeddings, word_index,

# train model
def train_model(model, train_data, val_data, word_index, batch_size, epochs=100, learning_rate=0.001):

    val_batch_size = batch_size

    # data
    X_train, y_train = train_data
    X_val, y_val = val_data

    # train metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.Accuracy(name='train_acc')
    train_recall = tf.keras.metrics.Recall(name='train_recall')
    train_precision = tf.keras.metrics.Precision(name='train_precision')

    # val metrics
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_acc = tf.keras.metrics.Accuracy(name='val_acc')
    val_recall = tf.keras.metrics.Recall(name='val_recall')
    val_precision = tf.keras.metrics.Precision(name='val_precision')

    # loss function and optimizer
    loss_fn = CategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # some metrics that help reduce learning rate on stagnation
    max_acc = 0
    val_thres = 3
    val_counter = 0

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                         tf.TensorSpec(shape=(None, 2), dtype=tf.float32)])

    def train_step(batch_X, batch_y):
        with tf.GradientTape() as tape:
            predictions = model(batch_X, training=True)
            loss = loss_fn(batch_y, predictions)  + tf.add_n(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, predictions

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                         tf.TensorSpec(shape=(None, 2), dtype=tf.float32)])

    def test_step(batch_X, batch_y):
        predictions = model(batch_X, training=False)
        loss = loss_fn(batch_y, predictions) +  tf.add_n(model.losses)
        return loss, predictions

    for epoch in range(epochs):

        # reduce learning rate and restore latest saved checkpoint
        if val_counter == val_thres:
            learning_rate = learning_rate / 10
            optimizer.lr.assign(learning_rate)
            print('learning rate changed to: {}'.format(learning_rate))
            val_counter = 0
            model.load_weights('sentiment_model.h5')

        # shuffle length keys to randomize train sequence
        train_keys = list(X_train.keys())
        val_keys = list(X_val.keys())
        random.shuffle(train_keys)
        random.shuffle(val_keys)

        # reset metrics

        train_loss.reset_states()
        val_loss.reset_states()

        train_acc.reset_states()
        val_acc.reset_states()

        train_recall.reset_states()
        val_recall.reset_states()

        train_precision.reset_states()
        val_precision.reset_states()

        # ------------------------ train epoch -------------------------

        for key in tqdm(train_keys):
            x_data = X_train[key]
            y_data = y_train[key]

            x_data, y_data = shuffle_lists(x_data, y_data)

            data_steps = np.ceil(len(x_data) / batch_size).astype(np.int16)

            for step in range(data_steps):

                batch_X = process_batch_sentences(
                    x_data[step * batch_size: (step+1) * batch_size], word_index)
                batch_y = y_data[step * batch_size: (step+1) * batch_size]

                loss, predictions = train_step(batch_X, batch_y)
                train_loss(loss)

                predictions = np.argmax(predictions, axis=1)
                batch_y = np.argmax(batch_y, axis=1)

                train_acc(batch_y, predictions)
                train_recall(batch_y, predictions)
                train_precision(batch_y, predictions)

        print('train loss is: {}'.format(train_loss.result()))
        print('train acc is: {}'.format(train_acc.result()))
        print('train recall is: {}'.format(train_recall.result()))
        print('train precision is: {}'.format(train_precision.result()))

        # ------------------------ val epoch ---------------------------

        for key in tqdm(val_keys):
            x_data = X_val[key]
            y_data = y_val[key]

            x_data, y_data = shuffle_lists(x_data, y_data)

            data_steps = np.ceil(len(x_data) / val_batch_size).astype(np.int16)

            for step in range(data_steps):

                batch_X = process_batch_sentences(
                    x_data[step * val_batch_size: (step+1) * val_batch_size], word_index)
                batch_y = y_data[step *
                                 val_batch_size: (step+1) * val_batch_size]

                loss, predictions = test_step(batch_X, batch_y)

                predictions = np.argmax(predictions, axis=1)
                batch_y = np.argmax(batch_y, axis=1)

                val_loss(loss)
                val_acc(batch_y, predictions)
                val_recall(batch_y, predictions)
                val_precision(batch_y, predictions)

        print('val loss is: {}'.format(val_loss.result()))
        print('val acc is: {}'.format(val_acc.result()))
        print('val recall is: {}'.format(val_recall.result()))
        print('val precision is: {}'.format(val_precision.result()))

        # save checkpoint if acc_value > max_acc

        accuracy = val_acc.result().numpy()
        precision = val_precision.result().numpy()
        recall = val_recall.result().numpy()

        # you can experiment with different acc value formulas.

        acc_value = accuracy  # * precision * recall

        if acc_value > max_acc:
            max_acc = acc_value
            model.save('sentiment_model.h5')
            print('model saved with acc = {}'.format(max_acc))
            val_counter = 0
        else:
            val_counter += 1

    return max_acc


# predict and create submission
def predict_model(model, predict_data, predict_indices, word_index, batch_size):

    with open('submission.csv', 'w') as f:
        print('Id,Prediction', file=f)

        test_keys = list(predict_data.keys())
        random.shuffle(test_keys)

        pred_list = [0] * 10000

        for key in tqdm(test_keys):
            x_data = predict_data[key]
            ind_data = predict_indices[key]

            data_steps = np.ceil(len(x_data) / batch_size).astype(np.int16)

            for step in range(data_steps):

                batch_X = process_batch_sentences(
                    x_data[step * batch_size: (step+1) * batch_size], word_index)
                batch_ind = ind_data[step * batch_size: (step+1) * batch_size]

                predictions = model(batch_X, training=False)

                predictions = np.argmax(predictions, axis=1)

                for pred, bind in zip(predictions, batch_ind):
                    if pred == 0:
                        pred = -1

                    pred_list[bind] = pred

        for find, fpred in enumerate(pred_list):
            print('{},{}'.format(find+1, fpred), file=f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=str, default='train',
                        choices=['train', 'test'], help='train or test step')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=512, help='batch_size')
    parser.add_argument('-vs', '--vocab_size', type=int,
                        default=10000, help='vocab_size')
    parser.add_argument('--rnn_dim', type=int,
                        default=512, help='rnn dimension')
    parser.add_argument('--dense1_dim', type=int,
                        default=1024, help='dense 1 dimension')
    parser.add_argument('--dense2_dim', type=int,
                        default=256, help='dense 2 dimension')
    parser.add_argument('-lr', '--learning_rate',
                    default=0.001, help='learning rate')
    parser.add_argument('--drop_rate', default=0.2, help='dropout rate')

    args = parser.parse_args()

    # possible steps: train, test
    step = args.step

    # vocab size max = 100000
    vocab_size = args.vocab_size

    # batch size
    batch_size = args.batch_size
    
    # hparams: rnn_dim, dense_1_dim, dense_2_dim, drop_rate
    hparams = args.rnn_dim, args.dense1_dim, args.dense2_dim, args.drop_rate

    data, embeddings, word_index = get_data(vocab_size, step)

    if step == 'train':
        model = create_model(embeddings, hparams, vocab_size)
        model.summary()
        # model.load_weights('sentiment_model_init.h5')
        train_data, val_data = data
        train_model(model, train_data, val_data, word_index,
                    batch_size, epochs=100, learning_rate=args.learning_rate)
    elif step == 'test':
        test_data, test_indices = data
        model = load_model('sentiment_model.h5')
        predict_model(model, test_data, test_indices, word_index, batch_size)
