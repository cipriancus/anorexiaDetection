from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from algorithm.helpers.task import Task
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import configparser
from random import shuffle
import pickle


class Clasify:
    def __init__(self):
        self.configs = configparser.ConfigParser()
        self.configs.read('model/configuration.cfg')

        self.use_saved_models = self.configs['Data']['use_saved_models']

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

        self.tfidf = self.get_idf()
        self.lda = self.get_latent_dirichlet_allocation()
        self.mlp = self.train()
        self.pipeline = self.get_pipeline()

    def get_idf(self):
        print('TFIDF')
        tfidf_file = self.configs['Feature'].get('tfidf', None)
        if tfidf_file and self.use_saved_models == 'true':
            print('loading vector file...')
            tfidf = pickle.load(open(tfidf_file, 'rb'))
        else:
            tfidf = TfidfVectorizer(analyzer='word', strip_accents='ascii', ngram_range=(1, 3),
                                    stop_words='english', max_features=3000, use_idf=True)
            tfidf.fit(self.posts)
            pickle.dump(tfidf, open('model/tfidf{}_{}.p'.format(self.task_number, self.split), 'wb'))
        print('DONE TFIDF')
        return tfidf

    def get_latent_dirichlet_allocation(self):
        print('LatentDirichletAllocation - LDA')
        lda_file = self.configs['Feature'].get('lda')
        if lda_file and self.use_saved_models == 'true':
            lda = pickle.load(open(lda_file, 'rb'))
        else:
            lda = LatentDirichletAllocation(n_components=30, learning_method='online')
            lda.fit(self.tfidf.transform(self.posts))
            pickle.dump(lda, open('model/lda_{}_{}.p'.format(self.task_number, self.split), 'wb'))
        print('LatentDirichletAllocation DONE.')
        return lda

    def train(self):
        print('MLP - TRAINING')
        mlp_file = self.configs['Model'].get('mlp')
        if mlp_file and self.use_saved_models == 'true':
            mlp = pickle.load(open(mlp_file, 'rb'))
        else:
            if bool(self.configs['Training']['undersample']):
                positives = list(filter(lambda s: s[0] == '1', zip(self.labels, self.users)))
                negatives = list(filter(lambda s: s[0] == '0', zip(self.labels, self.users)))
                shuffle(negatives)
                both = positives + negatives[:len(positives)]
                shuffle(both)
                self.labels, self.users = map(list, zip(*both))

            mlp = MLPClassifier(hidden_layer_sizes=(60, 30), max_iter=1500, activation='identity', solver='adam')
            # mlp = MLPClassifier(hidden_layer_sizes=(30, 60), max_iter=1500, activation='identity', solver='adam')
            # mlp = MLPClassifier(hidden_layer_sizes=(100, 150), max_iter=5000, activation='identity', solver='adam', learning_rate='adaptive')
            mlp.fit(self.lda.transform(self.tfidf.transform(self.users)), self.labels)
            pickle.dump(mlp, open('model/mlp_{}_{}.p'.format(self.task_number, self.split), 'wb'))
        print('MLP - TRAINING DONE')
        return mlp

    def get_pipeline(self):
        print('PIPELINE')
        pipeline = Pipeline([('tf', self.tfidf), ('lda', self.lda), ('mlp', self.mlp)])
        pickle.dump(pipeline, open('model/pipeline_{}_{}.p'.format(self.task_number, self.split), 'wb'))
        print('PIPELINE DONE')
        return pipeline

    def test(self):
        training = self.task.get_split(self.configs['Data']['split'], part='test', chunks=10)
        _, test_y, test_x = map(list, zip(*training))

        test_x = [' '.join(user) for user in test_x]
        pred_y = self.pipeline.predict(test_x)

        print(self.pipeline.predict_proba(test_x))

        print(classification_report(test_y, pred_y))

        print(confusion_matrix(test_y, pred_y))


if __name__ == "__main__":
    classifier = Clasify()
    classifier.test()
