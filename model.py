import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
# from load import mnist
from Dropout import Dropout
from GraphConvolution import GraphConvolution
import theano.tensor.nnet as nnet
from TemporalInputFeatures import TemporalInputFeatures
from updates import *
from six.moves import cPickle
import graphviz

class GCN(object):

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['training_set']
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.training_set = cPickle.load(open(self.training_set_file, 'rb'))

    def __init__(self,input_dim,hidden1,output_dim,learning_rate,weight_decay,drop_value=None,sparse_inputs=False,update_type=RMSprop(),clipnorm=0.0):
        
        self.X = T.matrix('X', dtype='float64')
        self.Y = T.matrix('Y',dtype='float64')
        self.adjacency = T.matrix('adjacency_matrix', dtype='float64')
        self.train_mask = T.vector('train_mask', dtype='float64')
        
        self.X.tag.test_value = np.random.rand(5, input_dim)
        self.Y.tag.test_value = np.random.rand(5, output_dim)
        self.adjacency.tag.test_value = np.random.rand(5, 5)

        self.update_type = update_type
        self.layers = []
        self.layers.append(TemporalInputFeatures(input_dim,self.X))
        self.layers.append(GraphConvolution(hidden1,self.adjacency,drop_value=drop_value))
        self.layers.append(GraphConvolution(output_dim,self.adjacency,activation_str='linear',drop_value=drop_value))
        # print(input_dim)
        # exit()


        L2_sqr = self.layers[0].L2_sqr
        for i in range(1,len(self.layers)):
            self.layers[i].connect(self.layers[i-1])
            L2_sqr += self.layers[i].L2_sqr 
        
        self.params = []
        for l in self.layers:
            if hasattr(l,'params'):
                self.params.extend(l.params)

        self.outputs = self.layers[-1].output()

        self.update_type.lr = learning_rate

        self.Y_pr = nnet.softmax(self.outputs)
        # self.cost = T.mean(nnet.categorical_crossentropy(self.Y_pr,self.Y) + weight_decay * L2_sqr)
        self.cost = self.masked_softmax_cross_entropy(self.Y_pr, self.Y, self.train_mask) + T.mean(weight_decay * L2_sqr)
        [self.updates,self.grads] = self.update_type.get_updates(self.params,self.cost)
        
        self.train = theano.function([self.X,self.Y,self.adjacency, self.train_mask],[self.cost,self.Y_pr],updates=self.updates,on_unused_input='ignore')
        self.norm = T.sqrt(sum([T.sum(g**2) for g in self.grads]))
        self.grad_norm = theano.function([self.X,self.Y,self.adjacency, self.train_mask],self.norm,on_unused_input='ignore')
        
    def masked_accuracy(self, Y_pred, Y, mask):
        """Accuracy with masking."""
        accuracy_all = np.equal(np.argmax(Y_pred,axis=1),np.argmax(Y,axis=1)).astype(float)
        # accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = mask.astype(float)
        mask /= np.mean(mask)
        accuracy_all *= mask
        return np.mean(accuracy_all)
        
    def accuracy(self,Y_pred,Y,mask):
        return self.masked_accuracy(Y_pred, Y, mask)
        # mask = mask.astype(float)
        # mask /= np.mean(mask)
        # print(Y[np.random.randint(0, Y.shape[0])])
        # print(Y_pred[np.random.randint(0, Y.shape[0])])
        # print(np.argmax(Y_pred,axis=1))
        # print(mask)
        # return np.mean(np.multiply(mask,np.equal(np.argmax(Y_pred,axis=1),np.argmax(Y,axis=1))))
        # return np.mean(np.equal(np.argmax(Y_pred,axis=1),np.argmax(Y,axis=1)))

    def masked_softmax_cross_entropy(self, preds, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = nnet.categorical_crossentropy(preds, labels)
        mask = mask.astype('float32')
        mask /= T.mean(mask)
        loss *= mask
        return T.mean(loss)

    def modelPass(self,X,Y,adjacency,train_mask, val_mask, test_mask,rng=np.random.RandomState(1234567890)):

        loss,Y_pr = self.train(X,Y,adjacency, train_mask)
        g = self.grad_norm(X,Y,adjacency, train_mask)
        trainAcc = self.accuracy(Y_pr,Y,train_mask)
        return loss, g, trainAcc

    def test(self, X, Y, adjacency, train_mask):
        _,Y_pr = self.train(X,Y,adjacency, train_mask)
        testAcc = self.accuracy(Y_pr,Y,train_mask)
        print("===============================================")
        print("Test Acc. " + str(testAcc))
        print("===============================================")
