import math
import sys

sys.path.append('..')

from collections import defaultdict

from bayes import Bayes


class MultiNominalBayes(Bayes):

    def load(self):
        super(MultiNominalBayes, self).load()
        self.likelihoods = defaultdict(lambda: {class_:0.5 for class_ in next(iter(self.likelihoods.items()))[1]}, self.likelihoods)

    def calculate_likelihoods(self, data):
        # data = {class1: [text1, text2, text3, text4, ...],
        #         class2: [text5, text6, text7, text8, ...]...}
        # text = 'I am programmer and I am student...'

        self.likelihoods = defaultdict(lambda: {class_:0.5 for class_ in data})
        word_frequencies_per_class = defaultdict(lambda: {class_:0 for class_ in data})
        total_words_per_class = defaultdict(lambda: 0)
        for class_ in data:
            for text in data[class_]:
                for word in self.get_words(text):
                    word_frequencies_per_class[word][class_] += 1
                    total_words_per_class[class_] += 1

        for word in word_frequencies_per_class:
            for class_ in data:
                self.likelihoods[word][class_] = (word_frequencies_per_class[word][class_]+1) / (total_words_per_class[class_]+2)


    def predict(self, text):

        # we do log because we are potentially dealing with relly small propabs
        # propability is same as self.prior just we computed log and create copy to not affect sotred dict
        propability_log = {class_: math.log(prior) for class_, prior in self.priors.items()}

        # likelihoods = {'word1': {class1: propab, class2: propab, ...},
        #                'word2': {class1: propab, class2: propab, ...},
        #                'word3': {class1: propab, class2: propab, ...},
        #                'word4': {class1: propab, class2: propab, ...}
        #                ...}
        for word in self.get_words(text):
            # here is_absent is used to cath if we are calculating probability of word
            # occurring in text or not occurring

            for class_, likelihood in self.likelihoods[word].items():
                propability_log[class_] += math.log(likelihood)

        return propability_log

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
    model = MultiNominalBayes()
    model.train(train_data, set_words, stop_words)
    model.load()

    compute(model, test_data)
