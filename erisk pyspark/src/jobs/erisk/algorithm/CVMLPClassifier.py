from src.jobs.erisk.algorithm.helpers.task import Task
from random import shuffle
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF, CountVectorizer, Tokenizer, StopWordsRemover
from pyspark.ml.clustering import LDA
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql import SQLContext
from sklearn.metrics import classification_report, confusion_matrix


class CVMLPClassifier:
    def __init__(self, sc, configs):
        self.sqlContext = SQLContext(sc)

        self.spark_context = spark_context

        self.configs = configs
        self.path_to_task = self.configs['Data']['task']
        self.undersample = self.configs['Training']['undersample']

        self.task = Task(self.path_to_task)
        self.task_number = self.path_to_task[-1]
        self.split = self.configs['Data']['split']
        self.training = self.task.get_split(self.split, part='train', chunks=10)
        _, self.labels, self.users = map(list, zip(*self.training))
        self.posts = [post for user in self.users for post in user]
        self.posts = list(filter(lambda p: len(p.split()) > 15, self.posts))
        self.labels, self.users = zip(*filter(lambda p: len(p[1]) > 10, zip(self.labels, self.users)))
        self.users = [' '.join(user) for user in self.users]

        if self.undersample != 'false':
            positives = list(filter(lambda s: s[0] == '1', zip(self.labels, self.users)))
            negatives = list(filter(lambda s: s[0] == '0', zip(self.labels, self.users)))
            shuffle(negatives)
            both = positives + negatives[:len(positives)]
            shuffle(both)
            self.labels, self.users = map(list, zip(*both))

        self.tokenizer = Tokenizer(inputCol="text", outputCol="rawWords")
        self.stopWords = StopWordsRemover(inputCol="rawWords", outputCol="words",caseSensitive=False,stopWords=StopWordsRemover.loadDefaultStopWords("english"))
        self.cv = CountVectorizer(inputCol="words", outputCol="rawFeatures",vocabSize=30000)
        self.idf = IDF(minDocFreq=2,inputCol="rawFeatures", outputCol="features")
        self.mlp = MultilayerPerceptronClassifier(maxIter=2000, layers=[30000, 80, 100, 2], blockSize=128, seed=1234)

        self.pipeline = Pipeline(stages=[self.tokenizer, self.stopWords, self.cv, self.idf, self.mlp])

        self.model = self.pipeline.fit(self.create_data_frame(self.users, self.labels))

    def create_data_frame(self, users, labels):
        if len(users) != len(labels):
            raise ValueError('The user list and labels are not the same')

        dataset = []
        for iterator in range(0, len(users)):
            dataset.append((users[iterator], int(labels[iterator])))

        return self.sqlContext.createDataFrame(dataset, ["text", "label"])

    def test(self):
        test = self.task.get_split(self.configs['Data']['split'], part='test', chunks=10)
        _, test_y, test_x = map(list, zip(*test))

        test_y = [int(value) for value in test_y]
        test_x = [' '.join(user) for user in test_x]

        dataset = []
        for user in test_x:
            dataset.append((user,))

        self.data = self.sqlContext.createDataFrame(dataset, ["text"])
        predict = self.model.transform(self.data)

        pred_y = [i.prediction for i in predict.select("prediction").collect()]

        print(confusion_matrix(test_y, pred_y))
        print(classification_report(test_y, pred_y))


if __name__ == "__main__":
    import configparser
    from pyspark import SparkContext
    from pyspark import SparkConf

    conf = SparkConf().setAppName("App")
    conf = (conf.setMaster('local[*]')
            .set('spark.executor.memory', '6G')
            .set('spark.driver.memory', '6G'))

    spark_context = SparkContext(conf=conf)

    sqlContext = SQLContext(spark_context)

    configs = configparser.ConfigParser()
    configs.read('../model/configuration.cfg')
    classifier = CVMLPClassifier(spark_context, configs)
    classifier.test()
