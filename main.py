
	# 'weight_decay' : T.scalar('weight_decay', dtype='float32'), 
	# 'drop_value' : T.scalar('drop_value', dtype='float32'),
	# 'adjacency' : T.matrix('adjacency_matrix', dtype='int32'),
	# 'X' : T.matrix('input', dtype=theano.config.floatX),
	# 'Y' : T.col('labels',dtype=theano.config.floatX),
	# 'std' : T.scalar('std', dtype=theano.config.floatX),
	# 'learning_rate' : T.scalar('learning rate', dtype=theano.config.floatX)

from __future__ import division
from __future__ import print_function
from theano import tensor as T
import time
import theano.sparse.basic as sp
from utils import *
from model import GCN
import  scipy.sparse as ssp
# Set random seed
seed = 123
np.random.seed(seed)

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora') # check if this is the one used
theano.config.exception_verbosity='high'

# Some preprocessing
features = preprocess_features(features)
adjacency = [preprocess_adj(adj)]

M = np.shape(y_train)[1]
weight_decay = 5e-4
learning_rate = 0.01
num_features_nonzero = features[1].shape[0]
drop_value = .5
std = 1e-5

a = ssp.coo_matrix((adjacency[0][1],(adjacency[0][0][:,0],adjacency[0][0][:,1])), shape=(adjacency[0][2]))
adjacency = a.todense()
a = ssp.coo_matrix((features[1],(features[0][:,0],features[0][:,1])), shape=(features[2]))
features = a.todense()
input_dim = (np.shape(features))[1]
model = GCN(input_dim,16,M,learning_rate,weight_decay,drop_value=drop_value)

print(np.shape(features))
print(np.shape(y_train))
print(np.shape(adjacency))
model.fitModel(features,y_train,adjacency)
