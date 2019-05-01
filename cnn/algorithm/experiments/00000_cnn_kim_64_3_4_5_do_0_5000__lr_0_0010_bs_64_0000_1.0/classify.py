from algorithm.helpers.task import Task
import configparser
from texcla import experiment, data
from texcla.models import TokenModelFactory, YoonKimCNN
from texcla.preprocessing import FastTextWikiTokenizer


# training = self.task.get_split(self.configs['Data']['split'], part='test', chunks=10)
# _, test_y, test_x = map(list, zip(*training))
#
# test_x = [' '.join(user) for user in test_x]
# pred_y = self.pipeline.predict(test_x)


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

        tokenizer = FastTextWikiTokenizer()

        # preprocess data (once)
        experiment.setup_data(self.users, self.labels, tokenizer, 'data.bin', max_len=100)

        # load data
        ds = data.Dataset.load('data.bin')

        # construct base
        factory = TokenModelFactory(
            ds.num_classes, ds.tokenizer.token_index, max_tokens=100,
            embedding_type='fasttext.wiki.simple', embedding_dims=300)

        # choose a model
        word_encoder_model = YoonKimCNN()

        # build a model
        model = factory.build_model(
            token_encoder_model=word_encoder_model, trainable_embeddings=False)

        # use experiment.train as wrapper for Keras.fit()
        experiment.train(x=ds.X, y=ds.y, validation_split=0.1, model=model,
                         word_encoder_model=word_encoder_model)


clasify = Clasify()
