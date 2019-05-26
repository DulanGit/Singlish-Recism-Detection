'''
Created on Apr 01, 2019

@author: dulan
'''
from params import DIR, MISC
import os.path as osp
from main.pickel_helper import PickelHelper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from main.classify import classify
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from params import FILES
import math


class Model(object):
    def __init__(self):
        self.model = None  # self.create_model(input_dim)
        self.bow_transformer = None
        self.tfidf_transformer = None
        self.pic_obj = PickelHelper()
        self.classify = classify()
        self.logger = None

    def create_model(self, input_dim):
        raise NotImplementedError

    def train(self, x_corpus, y_corpus):
        raise NotImplementedError

    def train_with_test(self, x_corpus, y_corpus, x_corpus_, y_corpus_):
        raise NotImplementedError

    def predict(self, text):
        raise NotImplementedError

    def save_model(self, model_name=''):
        self.model.save_weights(osp.join(DIR.DEF_SAV_LOC, model_name))

    def load_model(self, weights_file=''):
        self.model.load_weights(osp.join(DIR.DEF_SAV_LOC, weights_file))

    def save_transformers(self, filename):
        transform = {'bow': self.bow_transformer, 'tfidf': self.tfidf_transformer}
        self.pic_obj.save_obj(filename, transform)

    def load_transformers(self, filename):
        transform = self.pic_obj.load_obj(filename)
        self.tfidf_transformer = transform['tfidf']
        self.bow_transformer = transform['bow']

    def train_feature_gen(self, msg_train):
        self.bow_transformer = CountVectorizer(analyzer=self.classify.text_process).fit(msg_train)
        messages_bow = self.bow_transformer.transform(msg_train)
        self.tfidf_transformer = TfidfTransformer().fit(messages_bow)
        return self.tfidf_transformer.transform(messages_bow)

    def trans_val(self, val):
        if val == MISC.CLASSES[0]:
            return [1, 0]
        else:
            return [0, 1]

    def get_features(self, data):
        bw_msg = self.bow_transformer.transform(data)
        tfidef_msg = self.tfidf_transformer.transform(bw_msg)
        return tfidef_msg

    def evaluate(self, X, Y):
        # evaluate the model
        scores = self.model.evaluate(X, Y)
        self.logger.info("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        return scores[1]

    def log_n_print(self, msg):
        self.logger.info(msg)
        print(msg)

    def test_accuracy(self, feature_generator, optim_dict):
        messages_test = pd.read_csv(FILES.SEP_CSV_FILE_PATHS.format('test'), sep=',', names=["message", "label"])
        test_x, test_y = messages_test['message'], messages_test['label']
        test_features = feature_generator(test_x, optim_dict)
        predictions = ["Racist" if np.argmax(pred) == 0 else "Neutral" for pred in self.model.predict(test_features)]
        mat = confusion_matrix(predictions, test_y)
        total_acc = 1.0*(mat[0][0]+mat[1][1])/(mat[0][0]+ mat[0][1]+ mat[1][0]+ mat[1][1])
        self.log_n_print("==== Actual ====\n\t\tclass 1\t class 2\nclass1\t{}\t{}\nclass2\t{}\t{}\nAccuracy{}"
              .format(mat[0][0], mat[0][1], mat[1][0], mat[1][1], total_acc))
        self.log_n_print(classification_report(predictions, test_y))
        return total_acc

    def load_csv(self, filename):
        messages = pd.read_csv(filename, sep=',', names=["message", "label"], squeeze=True)
        new_messages = {"message":[], "label":[]}
        for i, msg in enumerate(messages["message"]):
            if not isinstance(msg, str):
                print("{} - removed {}".format(msg, type(msg)))
            else:
                new_messages["message"].append(msg)
                new_messages["label"].append(messages["label"][i])

        return new_messages
