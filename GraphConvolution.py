from __future__ import print_function
from __future__ import division
import theano
import numpy as np
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sparse.basic as sp
from theano.tensor.elemwise import CAReduce
from activations import *
from inits import *
from utils import *
from Dropout import Dropout



class GraphConvolution(object):
	
	def __init__(self, size,adjacency, num_features_nonzero=False, drop_value=None,rng=None, init='glorot',bias=False,sparse_inputs=False,dropout=True, activation_str='rectify', weights=False,featureless=False):

		self.settings = locals()
		del self.settings['self']
		self.sparse_inputs = sparse_inputs
		self.size = size
		self.rng = rng
		temp = inits()
		self.init = getattr(temp,init)
		temp = activations()
		self.activation = getattr(temp,activation_str)
		self.featureless = featureless
		self.weights = weights
		self.bias = bias
		self.adjacency = adjacency
		if dropout:
			self.drop_value = drop_value
		else:
			self.drop_value = 0	
			
		if self.sparse_inputs:
			self.num_features_nonzero = num_features_nonzero

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size
		# self.W = list()
		# for i in range(len(self.adjacency)):
		# 	self.W.append(self.init((self.inputD,self.size),rng=self.rng))
		self.W = self.init((self.inputD,self.size),rng=self.rng)
		
		if self.bias:
			self.b = zero0s((self.size))
			self.params = [self.W, self.b]
		else:
			self.params = [self.W]	

		if (self.weights):
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))

		# for i in range(len(self.W)):
		# 	self.L2_sqr = (self.W[i] ** 2).sum()
		self.L2_sqr = (self.W ** 2).sum()


	def output(self,seq_output=True):
		x = self.layer_below.output(seq_output=seq_output)
		# return self.activation(T.dot(X, self.W) + self.b)		

 	# 	dropout = Dropout()
		# if self.sparse_inputs:
		# 	x = dropout.sparse_dropout(x, self.drop_value, self.num_features_nonzero) # Not written completely
		# else:
		# 	x = dropout.dropout_layer(x, self.drop_value)#, train)

		# convolve
		# theano.shared(value=np.zeros(shape,dtype=theano.config.floatX))
		# output = zero0s((self.inputD,self.size))
		supports = list()
		# for i in range(len(self.adjacency)):

		if not self.featureless:
			if self.sparse_inputs:
				pre_sup = sp.dot(x, self.W)
			else:
				pre_sup = T.dot(x, self.W)
		else:
			pre_sup = self.W[i]
		support = T.dot(self.adjacency, pre_sup) #5.5 5.10x10.5
		supports.append(support)
		output = support

		if self.bias:
			output += self.b
		return self.activation(output) 