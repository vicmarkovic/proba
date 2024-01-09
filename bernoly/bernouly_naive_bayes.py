import json
import math
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.metrics import roc_curve, auc

import sys

from bayes import Bayes

sys.path.append('..')


class BernoulliBayes(Bayes):

    def preprocess_data(self, data):
        '''
        it removes duplicate words in each text(in these case email) by creating a set of that
        text that is splitted by space
        :return:
        '''
        # preprocessd_data will be same as data just instead of text it will contain set of words

        # before
        # data = {class1: [text1, text2, text3, text4, ...],
        #         class2: [text5, text6, text7, text8, ...]...}
        # text = 'I am programmer and I am student...'

        preprocessed_data = {}
        for class_, texts in data.items():
            preprocessed_data[class_] = []
            for text in texts:
                preprocessed_data[class_].append(set(self.get_words(text)))

        # after
        # preprocessed_data = {class1: [text1, text2, text3, text4, ...],
        #         class2: [text5, text6, text7, text8, ...]...}
        # text = {'I', 'am', 'programmer', 'and', 'student', ...}

        return preprocessed_data

    def calculate_likelihoods(self, data):
        # likelihoods = {'word1': {class1: propab, class2: propab, ...},
        #                'word2': {class1: propab, class2: propab, ...},
        #                'word3': {class1: propab, class2: propab, ...},
        #                'word4': {class1: propab, class2: propab, ...}
        #                ...}

        for class_, texts in data.items():
            # texts = [text1, text2, text3]
            for text in texts:
                # text = {'I', 'am', 'programmer', 'and', 'student', ...}
                for word in text:
                    if word not in self.likelihoods:
                        # we do laplace smoothing
                        self.likelihoods[word] = {class__: 1 / (len(data[class__]) + 1) for class__ in data}

                    self.likelihoods[word][class_] += 1 / (len(data[class_]) + 1)
        return self.likelihoods

    def predict(self, text):
        # we can create words just by these we do not need to filter them because we
        # are iterating through our vocab and checking if it occurred in text
        words = set(text.split())

        # we do log because we are potentially dealing with relly small propabs
        # propability is same as self.prior just we computed log and create copy to not affect sotred dict
        propability_log = {class_: math.log(prior) for class_, prior in self.priors.items()}

        # likelihoods = {'word1': {class1: propab, class2: propab, ...},
        #                'word2': {class1: propab, class2: propab, ...},
        #                'word3': {class1: propab, class2: propab, ...},
        #                'word4': {class1: propab, class2: propab, ...}
        #                ...}
        for word in self.likelihoods:
            # here is_absent is used to cath if we are calculating probability of word
            # occurring in text or not occurring
            is_absent = 0
            if word not in words:
                is_absent = 1
            for class_, likelihood in self.likelihoods[word].items():
                propability_log[class_] += math.log(abs(is_absent - likelihood))

        # here we calculate actual propabilty not just log
        propability = {}
        for class_ in propability_log:
            propability[class_] = 1 / sum(
                [
                    math.exp(propability_log[class__] - propability_log[class_])
                    for class__ in propability_log
                ]
            )
        return propability_log, propability


if __name__ == '__main__':
    from nltk.corpus import words
    from nltk.corpus import stopwords
    import nltk
    import data
    from metrics import *

    nltk.download('stopwords')
    set_words = set(words.words())
    stop_words = set(stopwords.words('english'))

    train_data, test_data = data.load()
    model = BernoulliBayes()
    model.train(train_data, set_words, stop_words)
    model.load()

    # histogram(model, train_data)
    compute(model, test_data)
    # sensitivity_specificity(model, test_data)
