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
import graphviz

class GCN(object):

    def __init__(self,input_dim,hidden1,output_dim,learning_rate,weight_decay,drop_value=None,sparse_inputs=False,update_type=RMSprop(),clipnorm=0.0):
        
        self.X = T.matrix('X', dtype='float64')
        self.Y = T.matrix('Y',dtype='float64')
        self.adjacency = T.matrix('adjacency_matrix', dtype='float64')
        
        self.X.tag.test_value = np.random.rand(5, 1433)
        self.Y.tag.test_value = np.random.rand(5, 7)
        self.adjacency.tag.test_value = np.random.rand(5, 5)

        self.update_type = update_type
        self.layers = []
        self.layers.append(TemporalInputFeatures(input_dim,self.X))
        self.layers.append(GraphConvolution(hidden1,self.adjacency,drop_value=drop_value))
        self.layers.append(GraphConvolution(output_dim,self.adjacency,activation_str='linear',drop_value=drop_value))


        for i in range(1,len(self.layers)):
            self.layers[i].connect(self.layers[i-1])
        
        self.params = []
        for l in self.layers:
            if hasattr(l,'params'):
                self.params.extend(l.params)

        self.outputs = self.layers[-1].output()

        self.update_type.lr = learning_rate

        self.Y_pr = nnet.softmax(self.outputs)
        theano.printing.pydotprint(self.Y_pr, outfile="pic.png", var_with_name_simple=True)
        self.cost = T.mean(nnet.categorical_crossentropy(self.Y_pr,self.Y) + weight_decay * self.layers[-1].L2_sqr)
        [self.updates,self.grads] = self.update_type.get_updates(self.params,self.cost)
        
        self.train = theano.function([self.X,self.Y,self.adjacency],[self.cost,self.Y_pr],updates=self.updates,on_unused_input='ignore')
        
        
        self.norm = T.sqrt(sum([T.sum(g**2) for g in self.grads]))
        self.grad_norm = theano.function([self.X,self.Y,self.adjacency],self.norm,on_unused_input='ignore')
        
        
    def accuracy(self,Y_pred,Y,mask):
        mask = mask.astype(float)
        mask /= np.mean(mask)
        print(Y[0])
        print(Y_pred[0])
        # print(np.argmax(Y_pred,axis=1))
        # print(mask)
        # return np.mean(np.multiply(mask,np.equal(np.argmax(Y_pred,axis=1),np.argmax(Y,axis=1))))
        return np.mean(np.equal(np.argmax(Y_pred,axis=1),np.argmax(Y,axis=1)))
    
    def fitModel(self,X,Y,adjacency,mask,epochs=10000,rng=np.random.RandomState(1234567890)):

        epoch_count = 0
        iterations = 0
        grad_norms = []
        loss_after_each_iter = []
        
        for i in range(epochs):
            loss,Y_pr = self.train(X,Y,adjacency)
            g = self.grad_norm(X,Y,adjacency)
            grad_norms.append(g)

            loss_after_each_iter.append(loss)
            termout = 'loss={0} iter={1} grad_norm={2} acc = {3}'.format(loss,i,g,self.accuracy(Y_pr,Y,mask))
            print(termout)

