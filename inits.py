# This method is taken from Passage https://github.com/IndicoDataSolutions/Passage

import numpy as np
import math
import theano
import theano.tensor as T
getattr
class inits():
	def uniform(self,shape, scale=0.05, rng=None):
		if rng is None:
			rng = np.random
		return theano.shared(value=rng.uniform(low=-scale,high=scale,size=shape).astype(theano.config.floatX))

	def allones(self,shape, scale=0.05, rng=None):
		if rng is None:
			rng = np.random	
		return theano.shared(value=np.ones(shape).astype(theano.config.floatX))

	def normal(self,shape, scale=0.05, rng=None):
		if rng is None:
			rng = np.random	
		return theano.shared(value=(rng.randn(*shape) * scale).astype(theano.config.floatX))

	def glorot(self,shape,rng=None):
		if rng is None:
			rng = np.random
		# print(shape)
		var = 6.0/(shape[0]+shape[1])
		stddev = math.sqrt(var)
		# return theano.shared(value=(rng.normal(0.0,stddev,size=shape)).astype(theano.config.floatX))
		return theano.shared(value=(rng.uniform(size=shape,low=-stddev,high=stddev)).astype(theano.config.floatX))

	def orthogonal(self,shape, scale=1.1, rng=None):
		""" benanne lasagne ortho init (faster than qr approach)"""
		if rng is None:
			rng = np.random	
		flat_shape = (shape[0], np.prod(shape[1:]))
		a = rng.normal(0.0, 1.0, flat_shape)
		u, _, v = np.linalg.svd(a, full_matrices=False)
		q = u if u.shape == flat_shape else v # pick the one with the correct shape
		q = q.reshape(shape)
		return theano.shared(value=(scale * q[:shape[0], :shape[1]]).astype(theano.config.floatX))
