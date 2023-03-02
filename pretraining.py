
"""Self-supervised model for contrastive learning task."""

# Script that pre-trains an encoder based on musical source association
# Usage: python3 pretraining.py datapath model_dir [--silent --mask] --sources source1 source2 ...
# datapath: where the pretraining dataset spectrograms (from both full songs and sources) are stored
# model_dir: path to the directory to store the model
# --silent: whether to use silent segments during pretraining (default - False, True if given)
# --mask: if given as an argument, reimplements the data-driven augmentation pipeline outlined in the S3T paper (Zhao et al, ICASSP-22)
#  + activates only if no sources are given
# --sources: list of sources to use (out of 'bass' 'drums', 'other', 'vocals'), might be empty.

import os
import sys
import numpy as np
import scipy as sp
import sklearn
import argparse
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import collections
import json
import tensorflow as tf
from tensorflow.keras import backend as K

from cola import constants,data
from mscol import network


def expand_labels(Y,n):
  Y = np.tile(Y,(n,1,1))
  Y = np.transpose(Y,(1,0,2))
  Yx = np.reshape(Y,(Y.shape[1]*Y.shape[0],Y.shape[2]))
  return Yx

def mask_and_shift(X):

  coins = np.random.uniform(0,1,4) #generates the probabilities that each of the random augmentations will be applied
  if coins[0] > 0.5:
    t = int(np.random.uniform(1,10))
    xs = (np.random.uniform(0,87,t)).astype(int) #time masking
    xm = (np.random.uniform(1,10,t)).astype(int)
    for i in range(0,t):
      X[xs[i]:xs[i]+xm[i],:] = 0
  if coins[1] > 0.5:
    f = int(np.random.uniform(1,5))
    xs = (np.random.uniform(0,64,f)).astype(int) #freq masking
    xm = (np.random.uniform(1,12,f)).astype(int)
    for i in range(0,f):
      X[:,xs[i]:xs[i]+xm[i]] = 0

  if coins[2] < 0.4:
    u = int(np.random.uniform(1,10)) #freq shifting (time shifting is redudant since the crops are already from diff. time)
    X[:,:64-u] = X[:,u:]
    X[:,64-u:] = 0

  #time warping
  if coins[3] < 0.4:
    tc = int(np.random.uniform(38,58))
    l = np.linspace(0,tc,49)
    u = np.linspace(tc+1,97,49)
    interp_space = np.concatenate((l,u),axis=0)
    Y = np.zeros((98,64,1))
    X = sp.interpolate.interp1d(x=np.arange(0,98),y=X,kind='linear',axis=0)(interp_space)
  return X

def _prepare_standard_example(x, is_training):
  """Creates an example for supervised training."""
  if is_training:
    s = int(np.random.uniform(0,x.shape[0]-98))
    x = x[s:s+98,:]
  else:
    x = tf.signal.frame(x,frame_length=98,frame_step=98,axis=0,pad_end=False)
  return x


def _prepare_example_sep(x,y):

  s = int(np.random.uniform(0,x.shape[0]-98))
  frames_anchors = x[s:s+98,:]
  s = int(np.random.uniform(0,x.shape[0]-98))
  frames_positives = y[s:s+98,:] 

  return frames_anchors, frames_positives


def _prepare_example(x,mask):
  """Creates an example (anchor-positive) for instance discrimination."""
  s = int(np.random.uniform(0,x.shape[0]-98))
  frames_anchors = x[s:s+98,:]

  s = int(np.random.uniform(0,x.shape[0]-98))
  frames_positives = x[s:s+98,:]

  if mask:
    frames_anchors = mask_and_shift(frames_anchors)
    frames_positives = mask_and_shift(frames_positives)

  return frames_anchors, frames_positives
    

parser = argparse.ArgumentParser()
parser.add_argument('datapath',type=str) #directory to load the data from
parser.add_argument('model_dir',type=str) #directory to store the model to
parser.add_argument('--silent',default=False,action='store_true') #whether to incorporate silent source segments
parser.add_argument('--mask',default=False,action='store_true') #if no sources are provided, mask reimplements the data-driven pipeline from Zhao et al.
parser.add_argument('--sources', nargs='+', default=[])

_mask = False
_src = False

args = parser.parse_args()
datapath = args.datapath
model_dir = args.model_dir
silent = args.silent

sources = args.sources
if len(sources) > 0:
  _src = True
else:
  _mask = args.mask


train_path = datapath+'/melspecs/train'
valid_path = datapath+'/melspecs/valid'
test_path = datapath+'/melspecs/test'

train_path_src = datapath+'/melspecs_src/train'
valid_path_src = datapath+'/melspecs_src/valid'

instr_keys = ['_bass.npy','_drums.npy','_other.npy','_vocals.npy']
idxs = []
for i in range(0,4): #select sources which will be used for model training.
  if instr_keys[i][1:-4] in sources:
    idxs.append(i)
valid_idxs = np.asarray(idxs)
src_train_numel = np.zeros((4,),dtype=np.int32)
train_keys_full = np.load(os.getcwd()+'/data-split/train_keys.npy')
train_keys_filt = [x[2:] for x in train_keys_full]
dict_ = os.listdir(train_path)
train_numel = len(dict_)
train_src = np.zeros((train_numel,len(instr_keys))) #binary matrix indicating the existence of src files.
train_keys = []
train_keys_src= [[],[],[],[]]
train_artist_keys = []
full_labels = np.load(os.getcwd()+'/data-split/y_train_pub.npy')
train_labels = np.zeros((train_numel,50))

ct = 0
for key in dict_:
  key_filt=key[:-11]
  try:
    idx = train_keys_filt.index(key_filt)
  except:
    continue
  ict = 0

  for ikey in instr_keys:
    train_src[ct,ict] = 0 #reinitialize.
    if os.path.isfile(train_path_src+'/'+key[:-9]+ikey):
      train_src[ct,ict] = 1
      train_keys_src[ict].append(key)
    ict += 1
  if (np.sum(train_src[ct,:]) > -1) or not _src:
    train_labels[ct,:] = full_labels[idx,:]
    train_keys.append(key)
    train_artist_keys.append(key.split('-')[0])
    ct+= 1
    
del dict_

train_numel = len(train_keys)
train_labels = train_labels[:train_numel,:]
train_src = train_src[:train_numel,:]
for i in range(0,4):
  src_train_numel[i] = len(train_keys_src[i])

src_valid_numel = np.zeros((4,),dtype=np.int32)
valid_keys_full = np.load(os.getcwd()+'/data-split/valid_keys.npy')
valid_keys_filt = [x[2:] for x in valid_keys_full]

dict_ = os.listdir(valid_path)
valid_numel = len(dict_)
valid_keys = []
valid_keys_src = [[],[],[],[]]
valid_artist_keys = []
valid_src = np.zeros((valid_numel,len(instr_keys)))
full_labels = np.load(os.getcwd()+'/data-split/y_valid_pub.npy')
valid_labels = np.zeros((valid_numel,50))
ct = 0
for key in dict_:
  key_filt=key[:-11]
  try:
    idx = valid_keys_filt.index(key_filt)
  except:
    continue
  ict = 0
  for ikey in instr_keys:
    valid_src[ct,ict] = 0
    if os.path.isfile(valid_path_src+'/'+key[:-9]+ikey):
      valid_src[ct,ict] = 1
      valid_keys_src[ict].append(key)
    ict += 1
  if (np.sum(valid_src[ct,:]) > -1) or not _src: 
    valid_labels[ct,:] = full_labels[idx,:]
    valid_keys.append(key)
    valid_artist_keys.append(key.split('-')[0])
    ct+=1 

del dict_

valid_numel = len(valid_keys)
valid_labels = valid_labels[:valid_numel,:]
valid_src = valid_src[:valid_numel,:]
for i in range(0,4):
  src_valid_numel[i] = len(valid_keys_src[i])    

test_keys_full = np.load(os.getcwd()+'/data-split/test_keys.npy')
test_keys_filt = [x[2:] for x in test_keys_full]
dict_ = os.listdir(test_path)
test_numel = len(dict_)
test_keys = []
full_labels = np.load(os.getcwd()+'/data-split/y_test_pub.npy')
test_labels = np.zeros((test_numel,50))
ct = 0

for key in dict_:

  key_filt=key[:-9]
  try:
    idx = test_keys_filt.index(key_filt)
  except:
    continue
  test_labels[ct,:] = full_labels[idx,:]
  ct+=1

  test_keys.append(key)

del dict_

test_numel = ct
test_labels = test_labels[:test_numel,:]    

stepinit = 0 
steps = 10000
contrastive_network = network.get_contrastive_network(
      embedding_dim=512,
      temperature=0.2,
      pooling_type='max',
      similarity_type=constants.SimilarityMeasure.BILINEAR)
contrastive_network.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],run_eagerly=True)

checkpoint = tf.train.Checkpoint(contrastive_network)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

batchSize = 8192
for i in range(stepinit, stepinit+steps):
  if (i % 10 == 0):
    print('training contrastive model...')
    running_loss = 0
    running_sca = 0

  full_comp = np.random.permutation(train_numel)
  batch_comp = full_comp[:batchSize] 
  
  anchors = np.empty((batchSize,98,64,1))
  positives = np.empty((batchSize,98,64,1))
  tids = np.random.choice(valid_idxs,size=64) #select a source task at random.
  if silent:
    ttypes = np.random.choice(2,size=64) #both applicable types of batch compositions
  else:
    ttypes = np.ones(64,) #only compose batches from active sources
  for n in range(0,batchSize):
    ttype = ttypes[n//128]
    tid = tids[n//128]
    if (n%128 == 0) and ttype == 1:
      bcomp = np.random.choice(len(train_keys_src[tid]),size=128,replace=False)
    if _src:
      if ttype == 0: #the source may be absent from the mixture (projection to null-space)
        id_ = np.where(train_src[batch_comp[n],:] == 1) #search for sources in the orig anchor.
        filename = train_path+'/'+train_keys[batch_comp[n]]
        y_raw = np.load(filename)
        if tid in id_[0]: 
          filename = train_path_src+'/'+train_keys[batch_comp[n]][:-9]+instr_keys[tid]
          ys = np.load(filename)
        else:
          ys = np.zeros((392,64,1)) #project to a nullspace.
      else: #we require that the source exists in the mixture
        sample = bcomp[n%128] 
        filename = train_path+'/'+train_keys_src[tid][sample]
        y_raw = np.load(filename)
        filename = train_path_src+'/'+train_keys_src[tid][sample][:-9]+instr_keys[tid]
        ys = np.load(filename)

    if _src:
      anchors[n,:,:,:], positives[n,:,:,:] = _prepare_example_sep(y_raw,ys)
    else:
      anchors[n,:,:,:],positives[n,:,:,:] = _prepare_example(y_raw,_mask) 
  
  data = np.concatenate((anchors,positives),axis=-1)
  evals = contrastive_network.fit(
    data,batch_size = 128,
    epochs=1,
    verbose=0,shuffle=False)
  running_loss += evals.history["loss"][0]
  running_sca += evals.history["sparse_categorical_accuracy"][0]
  
  if (i % 10 == 9): #evaluate the proxy task every 10 passes
    
    print('training contrastive task loss at step', i, ' ', running_loss/10)
    print('training contrastive task accuracy at step', i, ' ', running_sca/10)


    print('checkpoint: evaluating contrastive model...')
      
    batch_comp = np.random.permutation(valid_numel)
    anchors = np.empty((valid_numel,98,64,1))
    positives = np.empty((valid_numel,98,64,1))
    tids = np.random.choice(valid_idxs,size=1+valid_numel//128)
    if silent:
      ttypes = np.random.choice(2,size=1+valid_numel//128)
    else:
      ttypes = np.ones((1+valid_numel//128,))
    for n in range(0,valid_numel):
      ttype = ttypes[n//128]
      tid = tids[n//128] 
      if (n%128==0) and ttype == 1:
        bctemp = np.random.permutation(src_valid_numel[tid])[:128]
          
      if _src:
        if ttype == 0:
          id_ = np.where(valid_src[batch_comp[n],:] == 1)
          filename = valid_path+'/'+valid_keys[batch_comp[n]]
          y_raw = np.load(filename)
          if tid in id_[0]:
            filename = valid_path_src+'/'+valid_keys[batch_comp[n]][:-9]+instr_keys[tid]
            ys = np.load(filename)
          else:
            ys = np.zeros((392,64,1))
        else:
          sample = bctemp[n%128]
          filename = valid_path+'/'+valid_keys_src[tid][sample]
          y_raw = np.load(filename)
          filename = valid_path_src+'/'+valid_keys_src[tid][sample][:-9]+instr_keys[tid]
          ys = np.load(filename)

        anchors[n,:,:,:],positives[n,:,:,:] = _prepare_example_sep(y_raw,ys)
        
      else: 
        anchors[n,:,:,:],positives[n,:,:,:] = _prepare_example(y_raw,_mask)

    data = np.concatenate((anchors,positives),axis=-1)
    evals = contrastive_network.evaluate(data,batch_size=128,verbose=0)

    print('validation contrastive task loss at step', i, ' ', evals[0])
    print('validation contrastive task accuracy at step', i, ' ', evals[1])        
      
  if (i%1000 == 999):
    save_path = checkpoint.save('./'+sys.argv[1]+'checkpoint')

  if (i == 4999):
    K.set_value(contrastive_network.optimizer.lr, 0.5*K.get_value(contrastive_network.optimizer.lr))
