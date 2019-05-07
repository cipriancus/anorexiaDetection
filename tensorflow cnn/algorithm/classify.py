from algorithm.helpers.task import Task
import configparser
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import random
import nltk
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import keras.backend as K
import pickle

class Clasify:
    def __init__(self):
        self.configs = configparser.ConfigParser()
        self.configs.read('model/configuration.cfg')

        self.path_to_task = self.configs['Data']['task']
        self.task = Task(self.path_to_task)
        self.task_number = self.path_to_task[-1]
        self.split = self.configs['Data']['split']
        self.training = self.task.get_split(self.split, part='train', chunks=10)
        _, self.labels, self.users = map(list, zip(*self.training))
        self.posts = [post for user in self.users for post in user]
        self.posts = list(filter(lambda p: len(p.split()) > 15, self.posts))
        self.labels, self.users = zip(*filter(lambda p: len(p[1]) > 10, zip(self.labels, self.users)))
        self.users = [' '.join(user) for user in self.users]

        test = self.task.get_split(self.configs['Data']['split'], part='test', chunks=10)
        _, test_labels, test_posts = map(list, zip(*test))

        test_posts = [' '.join(user) for user in test_posts]

        random.seed(1000)

        wpt = nltk.WordPunctTokenizer()
        tokenized_corpus = [wpt.tokenize(document) for document in self.users]
        posts = tokenized_corpus

        # Set values for various parameters
        feature_size = 100  # Word vector dimensionality
        window_context = 50  # Context window size
        min_word_count = 5  # Minimum word count
        sample = 1e-3  # Downsample setting for frequent words

        ft_model = pickle.load(open('fast_text.p', 'rb'))
        # sg decides whether to use the skip-gram model (1) or CBOW (0)
        # ft_model = FastText(tokenized_corpus, size=feature_size, window=window_context,
        #                     min_count=min_word_count, sample=sample, sg=1, iter=50)
        # pickle.dump(ft_model, open('fast_text.p'.format(self.task_number, self.split), 'wb'))
        post_vectors = ft_model.wv
        vector_train_size = 100

        wpt = nltk.WordPunctTokenizer()
        tokenized_test_corpus = [wpt.tokenize(document) for document in test_posts]
        posts_test = tokenized_test_corpus

        ft_test_model = pickle.load(open('fast_text_test.p', 'rb'))
        # ft_test_model = FastText(tokenized_test_corpus, size=feature_size, window=window_context,
        #                          min_count=min_word_count, sample=sample, sg=1, iter=50)
        # pickle.dump(ft_test_model, open('fast_text_test.p'.format(self.task_number, self.split), 'wb'))
        test_posts_vectors = ft_test_model.wv
        vector_test_size = 100

        batch_size = 500
        no_epochs = 100
        max_no_tokens = 15

        model = Sequential()

        model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same',
                         input_shape=(max_no_tokens, vector_train_size)))
        model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
        model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=3))

        model.add(Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.3)))

        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.25))

        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

        tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)

        train_size = len(posts)
        test_size = len(posts_test)

        max_no_tokens = 15

        indexes = set(np.random.choice(len(posts), train_size, replace=False))
        indexes_test = set(np.random.choice(len(posts_test), test_size, replace=False))

        x_train = np.zeros((train_size, max_no_tokens, vector_train_size), dtype=K.floatx())
        y_train = np.zeros((train_size, 2), dtype=np.int32)

        x_test = np.zeros((test_size, max_no_tokens, vector_test_size), dtype=K.floatx())
        y_test = np.zeros((test_size, 2), dtype=np.int32)

        for i, index in enumerate(indexes):
            for t, token in enumerate(posts[index]):
                if t >= max_no_tokens:
                    break

                if token not in post_vectors:
                    continue

                    x_train[i, t, :] = post_vectors[token]

            if i < train_size:
                y_train[i, :] = [1.0, 0.0] if posts[index] == 0 else [0.0, 1.0]

        for i, index in enumerate(indexes_test):
            for t, token in enumerate(posts_test[index]):
                if t >= max_no_tokens:
                    break

                if token not in test_posts_vectors:
                    continue

                if i < test_size:
                    x_test[i, t, :] = test_posts_vectors[token]

            if i < test_size:
                y_test[i, :] = [1.0, 0.0] if posts_test[index] == 0 else [0.0, 1.0]

        model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=no_epochs,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard, EarlyStopping(min_delta=0.0001, patience=3)])

        print(model.metrics_names)

        print(model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1))

        model.save('cnn.model')


clasify = Clasify()
