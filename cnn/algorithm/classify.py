from algorithm.helpers.task import Task
import configparser


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

        training = self.task.get_split(self.configs['Data']['split'], part='test', chunks=10)
        _, test_y, test_x = map(list, zip(*training))

        test_x = [' '.join(user) for user in test_x]

        from nltk.tokenize import RegexpTokenizer
        from nltk.stem import WordNetLemmatizer
        from tqdm import tqdm
        import random
        import nltk

        random.seed(1000)

        lemmatizer = WordNetLemmatizer()
        tokenizer = RegexpTokenizer('[a-zA-Z0-9]\w+')

        print('Tokenizing ..')
        posts = [tokenizer.tokenize(post.lower()) for post in self.users]

        print('Done.')

        print('Lemmatizing ..')

        for tweet in self.users:
            lemmatized = [lemmatizer.lemmatize(word) for word in tweet]
            posts.append(lemmatized)

        vector_size = 256
        window = 5

        from gensim.test.utils import common_texts, get_tmpfile
        from gensim.models import Word2Vec

        import time

        word2vec_model = 'word2vec.model'

        print('Generating Word2Vec Vectors ..')

        start = time.time()

        model = Word2Vec(sentences=posts, size=vector_size, window=window, negative=20, iter=50, workers=4)

        print('Word2Vec Created in {} seconds.'.format(time.time() - start))

        model.save(word2vec_model)
        print('Word2Vec Model saved at {}'.format(word2vec_model))

        # Got to clear the memory!
        del model

        model = Word2Vec.load(word2vec_model)

        x_vectors = model.wv
        del model

        len(self.labels), len(posts)

        batch_size = 500
        no_epochs = 100
        max_no_tokens = 15

        from keras.models import Sequential
        from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping, TensorBoard

        model = Sequential()

        model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same',
                         input_shape=(max_no_tokens, vector_size)))
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

        model.summary()

        import numpy as np
        import keras.backend as K

        train_size = int(0.9 * (len(posts)))
        test_size = int(0.1 * (len(posts)))

        max_no_tokens = 15

        indexes = set(np.random.choice(len(posts), train_size + test_size, replace=False))

        x_train = np.zeros((train_size, max_no_tokens, vector_size), dtype=K.floatx())
        y_train = np.zeros((train_size, 2), dtype=np.int32)

        x_test = np.zeros((test_size, max_no_tokens, vector_size), dtype=K.floatx())
        y_test = np.zeros((test_size, 2), dtype=np.int32)

        for i, index in enumerate(indexes):
            for t, token in enumerate(posts[index]):
                if t >= max_no_tokens:
                    break

                if token not in x_vectors:
                    continue

                if i < train_size:
                    x_train[i, t, :] = x_vectors[token]
                else:
                    x_test[i - train_size, t, :] = x_vectors[token]

            if i < train_size:
                y_train[i, :] = [1.0, 0.0] if posts[index] == 0 else [0.0, 1.0]
            else:
                y_test[i - train_size, :] = [1.0, 0.0] if posts[index] == 0 else [0.0, 1.0]

        model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=no_epochs,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard, EarlyStopping(min_delta=0.0001, patience=3)])

        print(model.metrics_names)

        print(model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1))

        model.save('cnn.model')


clasify = Clasify()
