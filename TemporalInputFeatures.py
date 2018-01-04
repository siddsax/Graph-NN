from __future__ import print_function
from __future__ import division
import theano
import numpy as np
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sparse.basic as sp
from theano.tensor.elemwise import CAReduce


class TemporalInputFeatures(object):
	'''
	Use this layer to input dense features for RNN
	dim = Time x Num_examples x Feature_dimension
	'''
	def __init__(self,size,X,weights=None,skip_input=False,jump_up=False):
		self.settings = locals()
		del self.settings['self']
		self.size=size
		self.input = X
		self.inputD=size
		self.params=[]
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))
		self.skip_input = skip_input
		self.jump_up = jump_up

	def output(self,seq_output=True):
		return self.input

