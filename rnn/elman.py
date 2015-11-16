import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict


class ElmanRNNModel(object):
    
    def __init__(self, hidden_dims, num_classes, vocab_size, embed_dims, context_size, load=False):
        """
        hidden_dims :: dimension of the hidden layer
        num_classes :: number of classes
        vocab_size :: number of word embeddings in the vocabulary
        embed_dims :: dimension of the word embeddings
        context_size :: word window context size
        """
        # parameters of the model

        # word embedding matrix
        self.emb = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (vocab_size+1, embed_dims)).astype(theano.config.floatX)
        )  # add one for PADDING at the end

        # weights of context window -> hidden layer
        self.Wx = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (embed_dims * context_size, hidden_dims)).astype(theano.config.floatX)
        )

        # weights of hidden -> hidden for recurrence
        self.Wh = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (hidden_dims, hidden_dims)).astype(theano.config.floatX)
        )

        # classification weights (hidden -> output)
        self.W = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (hidden_dims, num_classes)).astype(theano.config.floatX)
        )

        # hidden unit bias weights?
        self.bh = theano.shared(numpy.zeros(hidden_dims, dtype=theano.config.floatX))

        # class bias weights?
        self.b = theano.shared(numpy.zeros(num_classes, dtype=theano.config.floatX))

        # initial hidden state
        self.h0 = theano.shared(numpy.zeros(hidden_dims, dtype=theano.config.floatX))

        # bundle
        self.params = [self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0]
        self.names = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        self._setup_functions(embed_dims, context_size)

    def _setup_functions(self, embed_dims, context_size):
        # This is the placeholder for the user-supplied word ids of a data minibatch
        idxs = T.imatrix()  # dimensions: (words in sentence/minibatch x context window size)

        # x is the input. Each idx is looked up in the embeddings matrix,
        # which results in an (words in sent x context window size x embedding size) matrix.
        # For the input layer, we reshape this by flattening the context word embeddings (concatenating them)
        # Then we have a (words in sent x (context window size * embedding size)) two dimensional array
        x = self.emb[idxs].reshape((idxs.shape[0], embed_dims*context_size))

        y = T.iscalar('y')  # the supplied training label placeholder

        def recurrence(x_t, h_tm1):
            # compute hidden layer at time t
            h_t = T.nnet.sigmoid(  # activation
                # project x_t (word + context window, i.e. (1 x embed_dims*context_size)) into the hidden layer
                # This results in a (1 x hidden_dims) vector representing the current words projection
                T.dot(x_t, self.Wx) +

                # We add the projection of the last timestep to the vector. h_tm1 is (1 x hidden_dims)
                T.dot(h_tm1, self.Wh) +

                # add the hidden bias term
                self.bh
            )
            # compute the output vector at time t. s_t has dimensions (1 x num_classes)
            s_t = T.nnet.softmax(  # softmax activation so it's a probability
                T.dot(h_t, self.W) +  # simply project the hidden representation we just computed into the output layer
                self.b  # and add the bias term
            )
            return [h_t, s_t]

        [h, s], _ = theano.scan(  # h and s should contain all the h_t's and s_t's we compute in recurrence
            fn=recurrence,
            sequences=x,  # we iterate over the rows (i.e. word by word)
            outputs_info=[self.h0, None],  # initialize loop with h0 (initial hidden state)
            n_steps=x.shape[0]  # number of words in the sentence
        )

        # s has three dimensions: the words (i.e. the t's), one dimension length 1, and the values of the
        p_y_given_x_lastword = s[-1, 0, :]  # this is the probability distribution of the last words' label
        p_y_given_x_sentence = s[:, 0, :]  # this is an array of probability distributions of EACH words' label
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        # loss function: we're aiming to DECREASE the NEGATIVE log probability of the label that is in fact correct
        nll = -T.log(p_y_given_x_lastword)[y]
        gradients = T.grad(nll, self.params)

        # This is one SGD update step
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))
        
        # compile the theano functions for train steps and classification
        self.train = theano.function(inputs=[idxs, y, lr],
                                     outputs=nll,
                                     updates=updates)

        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        # compile an additional normalization function which keeps the word embeddings inside the unit circle
        self.normalize = theano.function(
            inputs=[],
            updates={
                self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')
            }
        )

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

    @classmethod
    def load(cls, folder):
        params = OrderedDict()

        for paramname in ["embeddings", "Wx", "Wh", "W", "bh", "b", "h0"]:
            params[paramname] = numpy.load(folder + "/" + paramname + ".npy")

        # we need these parameters for initializing the theano functions
        embed_dims = params["embeddings"].shape[1]
        context_size = params["Wx"].shape[0] / embed_dims

        rnn = cls.__new__(cls)

        rnn.params = []
        rnn.names = []
        for paramname, param in params.iteritems():
            rnn.names.append(paramname)
            if paramname == "embeddings":
                paramname = "emb"

            param = theano.shared(param)
            setattr(rnn, paramname, param)
            rnn.params.append(getattr(rnn, paramname))

        rnn._setup_functions(embed_dims, context_size)

        return rnn
