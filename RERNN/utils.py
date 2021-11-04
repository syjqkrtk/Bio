from collections import defaultdict


class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.index_to_data = []
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.zerodata = [0,0,0,0,0,0,0,0,0]
        self.add_word(self.unknown, self.zerodata, count=0)

    def add_word(self, word, data, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            self.index_to_data.append(data)
        self.word_freq[word] += count

    def construct(self, words, datas):
        for i, word in enumerate(words):
            self.add_word(word, datas[i])

        self.total_words = float(sum(self.word_freq.values()))
        print('{} total words with {} uniques'.format(self.total_words, len(self.word_freq)))

    def encode(self, word):
        assert type(word) is str
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def get_data_from_index(self, index):
        return self.index_to_data[index]

    def get_data(self):
        return self.index_to_data

    def __len__(self):
        return len(self.word_freq)
