from __future__ import division
from __future__ import print_function
from theano import tensor as T
import time
import argparse
import theano.sparse.basic as sp
from utils import *
from gcn import GCN
import  scipy.sparse as ssp
from six.moves import cPickle
import cPickle
import copy_reg
import types
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--t', dest='train', type=int, default=1)
parser.add_argument('--dv', dest='dropValue', type=float, default=.5)
parser.add_argument('--lr', dest='learningRate', type=int, default=0.01)
parser.add_argument('--wd', dest='weightDecay', type=float, default=5e-4)
parser.add_argument('--ds', dest='dataSet', type=str, default='cora')
parser.add_argument('--lm', dest='loadModel', type=str, default='')
parser.add_argument('--me', dest='maxEpochs', type=int, default=10000)
parser.add_argument('--hd', dest='hiddenDim', type=int, default=16)
parser.add_argument('--ss', dest='saveStep', type=int, default=500)

args = parser.parse_args()
print(args)

seed = 123
np.random.seed(seed)

# Theano Options with Error Checking ON. Turn off some of these options for faster running
theano.config.exception_verbosity='high'
theano.config.optimizer='None'
theano.config.compute_test_value = 'warn'

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataSet) # check if this is the one used

# Some preprocessing
features = preprocess_features(features)
adjacency = [preprocess_adj(adj)]

# Making features and adjacency matrix dense
adj = ssp.coo_matrix((adjacency[0][1],(adjacency[0][0][:,0],adjacency[0][0][:,1])), shape=(adjacency[0][2]))
adjacency = adj.todense()
ft = ssp.coo_matrix((features[1],(features[0][:,0],features[0][:,1])), shape=(features[2]))
features = ft.todense()

M = np.shape(y_train)[1]
inputDim = (np.shape(features))[1]

# Graph CNN Model
model = GCN(inputDim,args.hiddenDim, np.shape(y_train)[1] ,args.learningRate,args.weightDecay,drop_value=args.dropValue)

if len(args.loadModel):
	f = open(args.loadModel, 'rb')
	model.params = cPickle.load(f)

if args.train:
	for i in range(args.maxEpochs):
		loss, g, trainAcc = model.modelPass(features,y_train,adjacency,train_mask, val_mask, test_mask)
		termout = 'Iter={1} | loss={0} grad_norm={2} Train Acc = {3} '.format(loss,i,g,trainAcc)
		print(termout)

		if i % args.saveStep == 0:
			save_file = open('gcn.th', 'wb')  # this will overwrite current contents
			cPickle.dump(model.params, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
			model.test(features, y_test, adjacency, test_mask)

else:
	print("+++++++")
	model.test(features, y_test, adjacency, test_mask)