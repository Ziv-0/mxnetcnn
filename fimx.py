import mxnet as mx
import glob
import cPickle
import numpy as np

def cifarread(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

tri_path='/home/ziv/Downloads/cifar-10-batches-py/train'
tes_path='/home/ziv/Downloads/cifar-10-batches-py/test_batch'
tra={'data':[],'labels':[]}
for c in glob.glob(tri_path+'/*'):
    dic=cifarread(c)
    tra['data'].append(dic['data'])
    tra['labels'].append(dic['labels'])


#
# tra=cifarread(tri_path)
tes=cifarread(tes_path)
#
tra['data']=np.array(tra['data']).reshape(-1,3,32,32)
tra['labels']=np.array(tra['labels']).reshape(50000)
tes['data']=np.array(tes['data']).reshape(-1,3,32,32)
tes['labels']=np.array(tes['labels'])

print tra['data'].shape
print tra['labels'].shape

batch_size=100
train_iter = mx.io.NDArrayIter(tra['data'], tra['labels'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(tes['data'], tes['labels'], batch_size)
#

data = mx.sym.var('data')
# data = mx.sym.reshape(data=data,shape=[3,32,32])
# # fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
# # act1 = mx.sym.Activation(data=fc1, act_type="relu")
# # fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
# # act2 = mx.sym.Activation(data=fc2, act_type="relu")
# # fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
# # mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
#
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=128)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=100)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# third conv layer
conv3 = mx.sym.Convolution(data=pool2, kernel=(3,3), num_filter=60)
tanh3 = mx.sym.Activation(data=conv3, act_type="tanh")
pool3 = mx.sym.Pooling(data=tanh3, pool_type="max", kernel=(2,2), stride=(2,2))
# fourth conv layer
conv4 = mx.sym.Convolution(data=pool3, kernel=(3,3), num_filter=60)
tanh4 = mx.sym.Activation(data=conv3, act_type="tanh")
pool4 = mx.sym.Pooling(data=tanh3, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool4)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=30)
tanh5 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh5, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
#
lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
#
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# # create a trainable module on CPU
# #lenet = mx.mod.Module(symbol=l, context=mx.cpu())
lenet_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=60)  # train for at most 10 dataset passes
prob = lenet_model.predict(val_iter)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
lenet_model.score(val_iter, acc)
print(acc)
