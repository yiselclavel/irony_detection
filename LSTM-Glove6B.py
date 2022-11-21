import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, Dropout
from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_core.python.keras.layers import LSTM, Flatten
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.saving import load_model

# gpu memory issues
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# session = tf.compat.v1.Session(config=config)

class IronyClassifier:
    # read the data
    dataframe = pd.read_csv('semeval train.csv', encoding='ISO-8859-1', sep=';;', engine='python')
    cols = ['text', 'label']
    dataframe.columns = cols
    X = dataframe.loc[:, 'text']
    Y = dataframe.loc[:, 'label']
    print(dataframe.shape)
    print(X.count())
    print(Y.count())
    Y = np.asarray(Y)

    # define weights with glove model
    embeddings_index = dict()
    words = []
    f = open('./glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split(' ')
        word = values[0]
        words.append(word)
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # define Tokenizer
    MAX_SEQUENCE_LENGTH = max([len(s.split()) for s in X])
    tokenizer_obj = Tokenizer(num_words=len(words))
    tokenizer_obj.fit_on_texts(words)
    word_index = tokenizer_obj.word_index
    vocab_size = len(word_index) + 1  # vocabulary size

    # construct embedding matrix
    EMBEDDING_DIM = 100
    embeddings_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, index in word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embeddings_vector = embeddings_index.get(word)
            if embeddings_vector is not None:
                embeddings_matrix[index] = embeddings_vector

    # pad sequences
    X_tokens = tokenizer_obj.texts_to_sequences(X)
    X_pad = pad_sequences(X_tokens, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    uniques, ids = np.unique(Y, return_inverse=True)
    y_train = tf.keras.utils.to_categorical(ids, len(uniques))
    print("Training data")
    print(X_pad.shape, Y.shape)

    # define 10-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    count = 0
    accuracy_list = list()
    precision_list = list()
    recall_list = list()
    f1_list = list()
    fig, ax = plt.subplots(figsize=(14, 8))

    for train, test in kfold.split(X_pad, Y):
        count += 1
        print('Fold: {0}'.format(count))
        # create model
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH, weights=[embeddings_matrix], trainable=True))
        model.add(LSTM(32, return_sequences=False, dropout=0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Flatten())
        model.add(Dense(2, activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # train the model
        print('Train...')
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
        mc = ModelCheckpoint('best_model' + str(count) + '.h5', monitor='val_accuracy', mode='max', verbose=1,
                             save_best_only=True)
        history = model.fit(X_pad[train], Y[train], validation_split=0.3, batch_size=64, epochs=20, verbose=2, callbacks=[es, mc])
        saved_model = load_model('best_model' + str(count) + '.h5')
        print()

        # evaluate the model
        y_predict = saved_model.predict_classes(X_pad[test])
        acc = accuracy_score(Y[test], y_predict)
        recall = recall_score(Y[test], y_predict, average='weighted')
        precision = precision_score(Y[test], y_predict, average='weighted')
        f1 = f1_score(Y[test], y_predict, average='weighted')
        print('Test accuracy:', acc)
        print('Test precision:', precision)
        print('Test recall:', recall)
        print('Test f1:', f1)
        accuracy_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        print(history.history.keys())
        plt.plot(history.history['accuracy'], color='blue')
        plt.plot(history.history['val_accuracy'], color='red')
        plt.title(str(count))
        plt.legend({"validation accuracy", "training accuracy"})
        plt.show()

    print('avg accuracy {0}'.format(np.average(accuracy_list)))
    print('avg precision {0}'.format(np.average(precision_list)))
    print('avg recall {0}'.format(np.average(recall_list)))
    print('avg f1 {0}'.format(np.average(f1_list)))
