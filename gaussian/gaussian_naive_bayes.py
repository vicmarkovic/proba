import json
import math

import sys
from collections import defaultdict

import numpy as np

from gaussian.metrics import compute

sys.path.append('..')

from bayes import Bayes


class GaussianBayes(Bayes):
    def preprocess_data(self, data):
        # before
        # data = {class1: [text1, text2, text3, text4, ...],
        #         class2: [text5, text6, text7, text8, ...]...}
        # text = 'I am programmer and I am student...'

        preprocessed_data = defaultdict(lambda: {class_: [] for class_ in data})
        self.n_texts_per_class = {class_: 0 for class_ in data}
        for class_ in data:
            for text in data[class_]:
                self.n_texts_per_class[class_] += 1
                seen_words_in_text = set()
                for word in self.get_words(text):
                    if word not in seen_words_in_text:
                        preprocessed_data[word][class_].append(0)
                        seen_words_in_text.add(word)
                    preprocessed_data[word][class_][-1] += 1
        # after
        # preprocessed_data = {word_1: {class1: {[freq_in_text_1, freq_in_text_2, ...freq_in_text_n],
        #                               class2: {[freq_in_text_1, freq_in_text_2, ...freq_in_text_n],}
        #                      word_2: {class1: {[freq_in_text_1, freq_in_text_2, ...freq_in_text_n],
        #                               class2: {[freq_in_text_1, freq_in_text_2, ...freq_in_text_n],}
        #                      word_3: {class1: {[freq_in_text_1, freq_in_text_2, ...freq_in_text_n],
        #                               class2: {[freq_in_text_1, freq_in_text_2, ...freq_in_text_n],}
        #                       }

        return preprocessed_data

    def get_words_freq(self, text):
        word_freqs = defaultdict(lambda: 0)
        for word in self.get_words(text):
            word_freqs[word] += 1
        return word_freqs

    def calculate_likelihoods(self, data):
        classes = data[''].keys()
        self.mean_per_class_per_word = defaultdict(lambda: {class_: 0.5 for class_ in classes})
        self.var_per_class_per_word = defaultdict(lambda: {class_: 0.5 for class_ in classes})
        for word in data:
            for class_, word_freqs_per_text in data[word].items():
                expanded_with_zeros = word_freqs_per_text + [0] * (
                        self.n_texts_per_class[class_] - len(word_freqs_per_text))
                mean = (np.sum(word_freqs_per_text) + 1) / (
                        self.n_texts_per_class[class_] + 2)

                var = (np.sum((np.array(expanded_with_zeros) - mean) ** 2) + 1) / (self.n_texts_per_class[class_] + 2)

                self.mean_per_class_per_word[word][class_] = mean
                self.var_per_class_per_word[word][class_] = var


    def predict(self, text):
        propability_log = {class_: math.log(prior) for class_, prior in self.priors.items()}
        for word, freq in self.get_words_freq(text).items():
            for class_ in propability_log:
                mean = self.mean_per_class_per_word[word][class_]
                var = self.var_per_class_per_word[word][class_]

                p = self.normal_dist(mean, var, freq)
                propability_log[class_] += p

        return propability_log

    def normal_dist(self, mean, var, freq):
        logpdf = -0.5 * ((freq - mean) / var) ** 2 - 0.5 * np.log(2 * np.pi * var ** 2)
        return logpdf

    def save(self):
        with open(self.save_path, 'w') as fp:
            json.dump([self.priors, self.mean_per_class_per_word, self.var_per_class_per_word, list(self.set_words), list(self.stop_words)], fp)

    def load(self):
        with open(self.save_path, 'r') as fp:
            self.priors, self.mean_per_class_per_word, self.var_per_class_per_word, self.set_words, self.stop_words = json.load(fp)
            self.set_words = set(self.set_words)
            self.stop_words = set(self.stop_words)

            self.mean_per_class_per_word = defaultdict(lambda: {class_: 0.5 for class_ in next(iter(self.mean_per_class_per_word.items()))[1]},
                                           self.mean_per_class_per_word)
            self.var_per_class_per_word = defaultdict(lambda: {class_: 0.5 for class_ in next(iter(self.var_per_class_per_word.items()))[1]},
                                           self.var_per_class_per_word)

if __name__ == '__main__':
    from nltk.corpus import words
    from nltk.corpus import stopwords
    import nltk
    import data

    nltk.download('stopwords')
    set_words = set(words.words())
    stop_words = set(stopwords.words('english'))

    train_data, test_data = data.load()
    model = GaussianBayes()
    model.train(train_data, set_words, stop_words)
    # model.load()
    compute(model, train_data)

