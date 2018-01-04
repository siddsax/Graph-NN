# This method is taken from Passage: 'https://github.com/IndicoDataSolutions/Passage'

import theano
import theano.tensor as T

class activations():
	def softmax(self,x):
	    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
	    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

	def rectify(self,x):
		return (x + abs(x)) / 2.0

	def tanh(self,x):
		return T.tanh(x)

	def sigmoid(self,x):
		return T.nnet.sigmoid(x)

	def linear(self,x):
		return x

	def t_rectify(self,x):
		return x * (x > 1)

	def t_linear(self,x):
		return x * (abs(x) > 1)

	def maxout(self,x):
		return T.maximum(x[:, 0::2], x[:, 1::2])

	def conv_maxout(self,x):
		return T.maximum(x[:, 0::2, :, :], x[:, 1::2, :, :])

	def clipped_maxout(self,x):
		return T.clip(T.maximum(x[:, 0::2], x[:, 1::2]), -5., 5.)

	def clipped_rectify(self,x):
		return T.clip((x + abs(x)) / 2.0, 0., 5.)

	def hard_tanh(self,x):
		return T.clip(x, -1. , 1.)

	def steeper_sigmoid(self,x):
		return 1./(1. + T.exp(-3.75 * x))

	def hard_sigmoid(self,x):
		return T.clip(x + 0.5, 0., 1.)
