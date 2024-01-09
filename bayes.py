import json


class Bayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.save_path = 'model.json'

    def preprocess_data(self, data):
        # before
        # data = {class1: [text1, text2, text3, text4, ...],
        #         class2: [text5, text6, text7, text8, ...]...}
        # text = 'I am programmer and I am student...'

        return data
    def get_words(self, text):
        return filter(self.filter_words, text.split())

    def filter_words(self, word):
        return word in self.set_words and word not in self.stop_words

    def train(self, data, set_words, stop_words):
        # data = {class1: [text1, text2, text3, text4, ...],
        #         class2: [text5, text6, text7, text8, ...]...}
        # text = 'I am programmer and I am student...'

        self.set_words = set_words  # valid words that our model will recognize
        self.stop_words = stop_words  # not significant words that we will omit

        self.calculate_priors(data)

        data = self.preprocess_data(data)

        self.calculate_likelihoods(data)

        self.save()

    def save(self):
        with open(self.save_path, 'w') as fp:
            json.dump([self.priors, self.likelihoods, list(self.set_words), list(self.stop_words)], fp)

    def load(self):
        with open(self.save_path, 'r') as fp:
            self.priors, self.likelihoods, self.set_words, self.stop_words = json.load(fp)
            self.set_words = set(self.set_words)
            self.stop_words = set(self.stop_words)

    def calculate_priors(self, data):
        '''
        we simply calculate prior for each class that is just propab of texts for some class
        divided by propab of all texts
        '''

        # texts is list that contains text
        # texts = [text1, text2, text3]
        # text = {'I', 'am', 'programmer', 'and', 'student', ...}
        len_per_class = {class_: len(texts)
                         for class_, texts in data.items()}

        # just summing up all texts i all classes, so just texts all together
        n_of_all_samples = sum([n for n in len_per_class.values()])

        # getting propability
        self.priors = {class_: len_per_class[class_] / n_of_all_samples for class_ in data}
