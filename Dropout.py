import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano

class Dropout():

	def __init__(self):
		np.random.seed(1234)
		self.rng = np.random.RandomState(1234)
		self.srng = RandomStreams(self.rng.randint(999999))

	def sparse_retain(self,x,dropout_mask):
		return x
	def sparse_dropout(self,x, keep_prob, noise_shape):

		random_tensor = keep_prob    
		random_tensor += self.srng.uniform(noise_shape)
		dropout_mask = np.floor(random_tensor).astype(bool)
		pre_out = sparse_retain(x, dropout_mask)
		return pre_out * (1./keep_prob)

	def dropit(self,weight, drop):
	
	    retain_prob = 1 - drop
	    mask = self.srng.binomial(n=1, p=retain_prob, size=weight.shape,dtype='floatX')
	    return theano.tensor.cast(weight * mask,theano.config.floatX)
	
	def dropout_layer(self,weight, drop, train = 1):
	    result = theano.ifelse.ifelse(theano.tensor.eq(train, 1),self.dropit(weight, drop), self.dont_dropit(weight, drop))
	    return result
	
	def dont_dropit(self,weight, drop):
		return (1 - drop)*theano.tensor.cast(weight, theano.config.floatX)