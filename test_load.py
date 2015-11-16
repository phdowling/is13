__author__ = 'dowling'
import numpy as np
from rnn.elman import ElmanRNNModel
from data import load
from utils.tools import contextwin
from metrics.accuracy import conlleval

CONTEXT_SIZE = 7
FOLDER = "elman_forward"

train_set, valid_set, test_set, dic = load.atisfold(3)
idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())


valid_lex, valid_ne, valid_y = valid_set
test_lex,  test_ne,  test_y = test_set


model = ElmanRNNModel.load("elman_forward")
# print model.params

predictions_test = [
    map(lambda x: idx2label[x],
        model.classify(np.asarray(contextwin(x, CONTEXT_SIZE)).astype('int32')))
    for x in test_lex
]

groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
words_test = [map(lambda x: idx2word[x], w) for w in test_lex]
predictions_valid = [
    map(
        lambda idx: idx2label[idx],
        model.classify(
            np.asarray(contextwin(x, CONTEXT_SIZE)).astype('int32'))
    )
    for x in valid_lex
]
groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]
# evaluation // compute the accuracy using conlleval.pl
res_test = conlleval(predictions_test, groundtruth_test, words_test, FOLDER + '/current.test.txt')
res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, FOLDER + '/current.valid.txt')

print res_test, res_valid