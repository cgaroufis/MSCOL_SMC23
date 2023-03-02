# Script that trains and evaluates a shallow classifier on top of a frozen pretrained encoder, for downstream tasks
# Usage: python3 downstream_filt.py path_to_model path_to_MTAT [--train]
# the model can be either used directly after pretraining or only evaluated
# --train, if given, trains the shallow classifier before evaluation
# if not, evaluates a trained encoder+shallow classifier on the given task

import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import cola
from cola import constants
from mscol import network

def prepare_standard_example(x, is_training):
    """Creates an example for supervised training."""
    if is_training:
        s = int(np.random.uniform(0,x.shape[0]-98))
        x = x[s:s+98,:]
    else:
        x = tf.signal.frame(x,frame_length=98,frame_step=98,axis=0,pad_end=False)
    return x

parser = argparse.ArgumentParser()
parser.add_argument('model_dir',type=str)
parser.add_argument('mtat_path',type=str)
parser.add_argument('--train',default=False,action='store_true')
args = parser.parse_args()
mtat_path = args.mtat_path
model_dir = args.model_dir
train = args.train


train_path = mtat_path+'/melspecs/train'
valid_path = mtat_path+'/melspecs/valid'
test_path = mtat_path+'/melspecs/test'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF


train_keys_full = np.load(os.getcwd()+'/data-split/train_keys.npy')
train_keys_filt = [x[2:] for x in train_keys_full]
print(train_keys_filt[0])
dict_ = os.listdir(train_path)
train_numel = len(dict_)
train_keys = []
full_labels = np.load(os.getcwd()+'/data-split/y_train_pub.npy')
train_labels = np.zeros((train_numel,50))
ct = 0

for key in dict_:
    key_filt=key[:-11]
    try:
        idx = train_keys_filt.index(key_filt)
    except:
        continue

    train_labels[ct,:] = full_labels[idx,:]
    train_keys.append(key)
    ct+=1

del dict_
train_numel = ct
train_labels = train_labels[:train_numel,:]
train_keys = train_keys[:train_numel]

valid_keys_full = np.load(os.getcwd()+'/data-split/valid_keys.npy')
valid_keys_filt = [x[2:] for x in valid_keys_full]

dict_ = os.listdir(valid_path)
valid_numel = len(dict_)
valid_keys = []
full_labels = np.load(os.getcwd()+'/data-split/y_valid_pub.npy')
valid_labels = np.zeros((valid_numel,50))
ct = 0
for key in dict_:
    key_filt=key[:-11]
    try:
        idx = valid_keys_filt.index(key_filt)
    except:
        continue

    valid_labels[ct,:] = full_labels[idx,:]
    ct+=1
    valid_keys.append(key)

del dict_
valid_numel = ct
valid_labels = valid_labels[:valid_numel,:]
valid_keys = valid_keys[:valid_numel]

test_keys_full = np.load(os.getcwd()+'/data-split/test_keys.npy')
test_keys_filt = [x[2:] for x in test_keys_full]
dict_ = os.listdir(test_path)
test_numel = len(dict_)
test_keys = []
full_labels = np.load(os.getcwd()+'/data-split/y_test_pub.npy')
test_labels = np.zeros((test_numel,50))
ct = 0

for key in dict_:
    a = True   
    key_filt=key[:-9] 
    try:
        idx = test_keys_filt.index(key_filt)
    except ValueError:
        a = False
    if a:
        test_labels[ct,:] = full_labels[idx,:]
        ct+=1
        test_keys.append(key)

del dict_
test_numel = ct
test_labels = test_labels[:test_numel,:]

contrastive_network = network.get_contrastive_network(
          embedding_dim=512,
          temperature=0.2,
          pooling_type='max',
          similarity_type=constants.SimilarityMeasure.BILINEAR)
contrastive_network.compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

contrastive_network.load_weights(tf.train.latest_checkpoint(sys.argv[1])).expect_partial()
encoder = contrastive_network.embedding_model.get_layer("encoder")

inputs = tf.keras.layers.Input(shape=(98, 64))
x = encoder(inputs) #pretrained encoder of the CSSL model
outputs = tf.keras.layers.Dense(50,activation='sigmoid')(x) #downstream model for tasks

model = tf.keras.Model(inputs, outputs)
model.get_layer("encoder").trainable = False #True for finetuning
model.compile(
         optimizer=tf.keras.optimizers.Adam(0.0005),
          loss=tf.keras.losses.BinaryCrossentropy(),
          metrics=[tf.keras.metrics.BinaryAccuracy()])

checkpoint = tf.train.Checkpoint(model)

print(train)

if train:
  
  best_val_acc = 0
  patience = 0
  batchSize = 8192
  epochs = 50

  for k in range(0,epochs):
    
    running_loss = 0
    running_acc = 0
    
    batchcomp = np.random.permutation(train_numel)
    batch_cnt = 0
    batchSize = 8192
    ct = 0
    
    for i in range(0,batchSize*(train_numel-batchSize)//batchSize,batchSize):

      batch_comp = batchcomp[i:i+batchSize]
      specs = np.empty((batchSize,98,64,1))
      for n in range(0,batchSize):
        filename = train_path+'/'+train_keys[batch_comp[n]]
        y_raw = np.load(filename) #loads a 4-sec excerpt
        specs[n,:,:,:] = prepare_standard_example(y_raw,True)

      labels = train_labels[batch_comp[:batchSize],:]
      evals = model.fit(x=specs,y=labels,batch_size=128,epochs=k+1,initial_epoch=k,verbose=0)

      running_loss += evals.history["loss"][0]
      running_acc += evals.history["binary_accuracy"][0]
      ct += 1
  
    running_loss = running_loss/ct
    running_acc = running_acc/ct
    print("Epoch",k+1,": Training Loss:", "{:.5f}".format(running_loss),"Training Accuracy:","{:.5f}".format(running_acc))

    batch_comp = np.random.permutation(valid_numel)
    specs = np.empty((valid_numel,98,64,1))
    for n in range(0,valid_numel):
      filename = valid_path+'/'+valid_keys[n]
      y_raw = np.load(filename)
      specs[n,:,:,:] = prepare_standard_example(y_raw,True)

    labels = valid_labels
    evals = model.evaluate(x=specs,y=labels,batch_size=128,verbose=0)
    print("Validation Loss:","{:.5f}".format(evals[0]),"Validation Accuracy","{:.5f}".format(evals[1]))
    if evals[1] > best_val_acc:
      best_val_acc = evals[1]
      patience = 0
      save_path = checkpoint.save('./'+sys.argv[1][:-1]+'_MTAT/'+'checkpoint')
    else:
      patience += 1
      if patience > 4:
        print('Epochs required for convergence of downstream model:',k-4)
        print('Performance of the downstream model in the training set', "{:.5f}".format(running_acc))
        print('Performance of the downstream model in the validation set', "{:.5f}".format(best_val_acc))
        break
    
model.load_weights(tf.train.latest_checkpoint(sys.argv[1][:-1]+'_MTAT/')).expect_partial()
evals_ = np.zeros((test_numel,50))
for kk in range(0,test_numel,361):
  specs = np.empty((29*361,98,64,1))
  ct = 0
  for n in range(kk,kk+361):
    filename = test_path+'/'+test_keys[n]
    y_raw = np.load(filename)
    specs[ct:ct+29,:,:,:] = prepare_standard_example(y_raw,False)
    ct += 29

  evals = model.predict(specs)
  evals = np.reshape(evals,(361,29,50))
  evals_[kk:kk+361,:] = np.sum(evals,axis=1)/29
          
print('roc-auc score',roc_auc_score(test_labels,evals_,average=None))
print('pr-auc score',average_precision_score(test_labels,evals_,average=None))
print('macro roc-auc score',roc_auc_score(test_labels,evals_))
print('macro pr-auc score',average_precision_score(test_labels,evals_))

