# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  LinearSVC

import logging
import sys

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)


log.info('source load')
test_source = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS'}

log.info('TaggedDocument')
test_sentences = TaggedLineSentence(test_source)

model = Doc2Vec.load('./imdb.d2v')

print(model.most_similar('good'))

test_arrays = numpy.zeros((25000, 150))
test_labels = numpy.zeros(25000)


for index, i in enumerate(test_sentences):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    feature = model.infer_vector(i[0])
    #print("printing feature")
    #print(feature)
    if index <12500:
        test_arrays[index] = feature
        test_labels[index] = 0
    else:
        test_arrays[index] = feature
        test_labels[index] = 1

    


