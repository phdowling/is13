import numpy
import time
import sys
import subprocess
import os
import random

from data import load
from rnn.elman import ElmanRNNModel
from metrics.accuracy import conlleval
from utils.tools import shuffle, minibatch, contextwin

LOAD = True

def main():
    settings = {
        'fold': 3,  # 5 folds 0,1,2,3,4
        'lr': 0.0627142536696559,
        'verbose': 1,
        'decay': False,  # decay on the learning rate if improvement stops
        'win': 7,  # number of words in the context window
        'bs': 9,  # number of backprop through time steps
        'nhidden': 100,  # number of hidden units
        'seed': 345,
        'emb_dimension': 100,  # dimension of word embedding
        'nepochs': 50
    }

    folder = os.path.basename(__file__).split('.')[0]

    if not os.path.exists(folder):
        os.mkdir(folder)

    # load the dataset
    train_set, valid_set, test_set, dic = load.atisfold(settings['fold'])
    idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex,  test_ne,  test_y = test_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    # instantiate the model
    numpy.random.seed(settings['seed'])
    random.seed(settings['seed'])

    if LOAD:
        print "Loading model from %s..." % folder

        rnn = ElmanRNNModel.load(folder)
    else:
        rnn = ElmanRNNModel(
            hidden_dims=settings['nhidden'],
            num_classes=nclasses,
            vocab_size=vocsize,
            embed_dims=settings['emb_dimension'],
            context_size=settings['win']
        )

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    settings['current_lr'] = settings['lr']
    for e in xrange(settings['nepochs']):
        # shuffle
        shuffle([train_lex, train_ne, train_y], settings['seed'])
        settings['current_epoch'] = e
        tic = time.time()
        for i in xrange(nsentences):
            cwords = contextwin(train_lex[i], settings['win'])

            words = map(
                lambda x: numpy.asarray(x).astype('int32'),
                minibatch(cwords, settings['bs'])
            )

            labels = train_y[i]

            for word_batch, label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, settings['current_lr'])
                rnn.normalize()

            if settings['verbose']:
                print '[learning] epoch %i >> %2.2f%%' % (e, (i+1)*100./nsentences), \
                    'completed in %.2f (sec) <<\r' % (time.time()-tic),
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [
            map(lambda x: idx2label[x],
                rnn.classify(numpy.asarray(contextwin(x, settings['win'])).astype('int32')))
            for x in test_lex
        ]

        groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y ]

        words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [
            map(
                lambda idx: idx2label[idx],
                rnn.classify(
                    numpy.asarray(contextwin(x, settings['win'])).astype('int32'))
            )
            for x in valid_lex
        ]

        groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]

        words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            if settings['verbose']:
                print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20
            settings['vf1'], settings['vp'], settings['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            settings['tf1'], settings['tp'], settings['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
            settings['be'] = e
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print ''

        # learning rate decay if no improvement in 10 epochs
        if settings['decay'] and abs(settings['be'] - settings['current_epoch']) >= 10:
            settings['current_lr'] *= 0.5

        if settings['current_lr'] < 1e-5:
            break

    print 'BEST RESULT: epoch', e, 'valid F1', settings['vf1'], 'best test F1', settings['tf1'], 'with the model', folder

if __name__ == '__main__':
    main()