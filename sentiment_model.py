from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Embedding, Dropout
from tensorflow.keras.layers import Activation, Layer, Softmax, Multiply, Lambda, Attention, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

# build and return model
def create_model(embeddings, hparams, vocab_size):

    word_dim = 300
    rnn_dim, dense_1_dim, dense_2_dim, drop_rate = hparams

    embeddings = embeddings[:vocab_size]

    input_seq = Input(shape=(None,))

    # replace word ids with word2vec embeddings
    embs = Embedding(input_dim=vocab_size, output_dim=word_dim,
                     weights=[embeddings], trainable=False)(input_seq)

    # ---------- first path -------------

    x1 = Bidirectional(LSTM(rnn_dim, return_sequences=True,
                            recurrent_dropout=drop_rate), merge_mode='concat')(embs)

    # bidirectional LSTM to parse sentence from both directions
    x1 = Bidirectional(LSTM(rnn_dim, return_sequences=True,
                            recurrent_dropout=drop_rate), merge_mode='concat')(x1)

    # attention mechanism
    attention = Dense(1)(x1)
    attention = Softmax(axis=1)(attention)
    context = Multiply()([attention, x1])
    p1 = Lambda(lambda x: K.sum(x, axis=1))(context)

    # ---------- second path ------------

    x2 = Dense(1024)(embs)
    x2 = Activation('relu')(x2)
    attention2 = Dense(1)(x2)
    attention2 = Softmax(axis=1)(attention2)
    context2 = Multiply()([attention2, p1])
    p2 = Lambda(lambda x: K.sum(x, axis=1))(context2)

    x = tf.concat([p1, p2], -1)

    if drop_rate > 0:
        x = Dropout(drop_rate)(x)

    x = Dense(dense_1_dim, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    if drop_rate > 0:
        x = Dropout(drop_rate)(x)

    x = Dense(dense_2_dim, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    if drop_rate > 0:
        x = Dropout(drop_rate)(x)

    x = Dense(2, kernel_initializer='he_normal')(x)
    x = Activation('softmax')(x)

    model = Model(input_seq, x)

    # weight decay example
    regularizer = tf.keras.regularizers.l2(0.001)
    for layer in model.layers:
        if isinstance(layer, Dense) or isinstance(layer, LSTM):
            model.add_loss(lambda layer=layer: regularizer(layer.kernel))

    return model
