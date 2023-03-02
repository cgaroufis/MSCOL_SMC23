# Script that trains and evaluates a shallow classifier on top of a frozen pretrained encoder, for downstream tasks
# Usage: python3 downstream_nsynth.py path_to_model path_to_NSynth [--train]
# the model can be either used directly after pretraining or only evaluated
# --train, if given, trains the shallow classifier before evaluation
# if not, evaluates a trained encoder+shallow classifier on the given task


import os
import argparse
import json
import collections #import Counter
import sys
import numpy as np
import tensorflow as tf
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import cola
from cola import constants
from mscol import network

def keys_to_labels(keys,labels):
    Y = np.zeros((len(keys),len(labels)))
    ctr = 0
    for instr in labels.keys():
        indices = [i for i, x in enumerate(keys) if x == instr]
        Y[indices,ctr] = 1
        ctr += 1

    return Y

def expand_labels(Y,n):
  Y = np.tile(Y,(n,1,1))
  Y = np.transpose(Y,(1,0,2))
  Yx = np.reshape(Y,(Y.shape[1]*Y.shape[0],Y.shape[2]))
  return Yx


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
parser.add_argument('nsynth_path',type=str)
parser.add_argument('--train',default=False,action='store_true')
args = parser.parse_args()
nsynth_path = args.nsynth_path
model_dir = args.model_dir
train = args.train

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()

train_path = nsynth_path+'/nsynth-train.jsonwav/nsynth-train'
valid_path = nsynth_path+'/nsynth-valid.jsonwav/nsynth-valid'
test_path = nsynth_path+'/nsynth-test.jsonwav/nsynth-test'

dict_ = json.loads(open(train_path+'/examples.json','r').read())
train_numel = len(dict_)
train_keys = []
train_instrs = []
for key in dict_:
    train_instrs.append(key.split('_')[0])
    train_keys.append(key)

del dict_

labels_cnt = collections.Counter(train_instrs)
print(labels_cnt)
train_labels = keys_to_labels(train_instrs,labels_cnt)
print('training labels',train_labels.shape)

dict_ = json.loads(open(valid_path+'/examples.json','r').read())
valid_numel = len(dict_)
valid_keys = []
valid_instrs = []
for key in dict_:
    valid_instrs.append(key.split('_')[0])
    valid_keys.append(key)

del dict_

valid_labels = keys_to_labels(valid_instrs,labels_cnt)
print('valid labels',valid_labels.shape)

dict_ = json.loads(open(test_path+'/examples.json','r').read())
test_numel = len(dict_)
test_keys = []
test_instrs = []
for key in dict_:
    test_instrs.append(key.split('_')[0])
    test_keys.append(key)

del dict_

test_labels = keys_to_labels(test_instrs,labels_cnt)

key = sys.argv[1].split('_')[0]
print(key)
print('test labels',test_labels.shape)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

contrastive_network = network.get_contrastive_network(
          embedding_dim=512,
          temperature=0.2,
          pooling_type='max',
          similarity_type=constants.SimilarityMeasure.BILINEAR)
contrastive_network.compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

contrastive_network.load_weights(tf.train.latest_checkpoint(model_dir)).expect_partial()
encoder = contrastive_network.embedding_model.get_layer("encoder")

inputs = tf.keras.layers.Input(shape=(98, 64))
x = encoder(inputs) #pretrained encoder of the CSSL model
x = tf.keras.layers.Dense(512,activation='relu')(x)
outputs = tf.keras.layers.Dense(11)(x) #downstream model for tasks

model = tf.keras.Model(inputs, outputs)
model.get_layer("encoder").trainable = False #True for finetuning
model.compile(
         optimizer=tf.keras.optimizers.Adam(0.0003),
          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.CategoricalAccuracy()])

checkpoint = tf.train.Checkpoint(model)


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
      batch_comp = batchcomp[batch_cnt:batch_cnt+batchSize]
      batch_cnt += batchSize

    
      specs = np.empty((batchSize,98,64,1))
      keys = [train_keys[x] for x in batch_comp]
      for n in range(0,batchSize):
        filename = train_path+'/melspecs/'+keys[n]+'.npy'
        y_raw = np.load(filename) #loads a 4-sec excerpt
        specs[n,:,:,:] = prepare_standard_example(y_raw,True)

      labels = train_labels[batch_comp[:batchSize],:]
      evals = model.fit(x=specs,y=labels,batch_size=128,epochs=k+1,initial_epoch=k,verbose=0)

      running_loss += evals.history["loss"][0]
      running_acc += evals.history["categorical_accuracy"][0]
      ct += 1
    
    running_loss = running_loss/ct
    running_acc = running_acc/ct
    print("Epoch",k+1,": Training Loss:", "{:.5f}".format(running_loss),"Training Accuracy:","{:.5f}".format(running_acc))

    batch_comp = np.random.permutation(valid_numel)
    specs = np.empty((valid_numel,98,64,1))
    for n in range(0,valid_numel):
      filename = valid_path+'/melspecs/'+valid_keys[n]+'.npy'
      y_raw = np.load(filename)
      specs[n,:,:,:] = prepare_standard_example(y_raw,True)

    labels = valid_labels
    evals = model.evaluate(x=specs,y=labels,batch_size=128,verbose=0)
    print("Validation Loss:","{:.5f}".format(evals[0]),"Validation Accuracy","{:.5f}".format(evals[1]))
    if evals[1] > best_val_acc:
      best_val_acc = evals[1]
      patience = 0
      save_path = checkpoint.save('./'+model_dir[:-1]+'_Nsynth/'+'checkpoint')
    else:
      patience += 1
      if patience > 4:
        print('Epochs required for convergence of downstream model:',k-4)
        print('Performance of the downstream model in the training set', "{:.5f}".format(running_acc))
        print('Performance of the downstream model in the validation set', "{:.5f}".format(best_val_acc))
        break


model.load_weights(tf.train.latest_checkpoint(model_dir[:-1]+'_Nsynth/')).expect_partial()
specs = np.empty((4*test_numel,98,64,1))
for n in range(0,4*test_numel,4):
  filename = test_path+'/melspecs/'+test_keys[n//4]+'.npy'
  y_raw = np.load(filename)
  specs[n:n+4,:,:,:] = prepare_standard_example(y_raw,False)

labels = expand_labels(test_labels,4)
evals = model.predict(specs)
evals = np.reshape(evals,(4096,4,11))
evals = np.sum(evals,axis=1)
evals_oh = (evals == np.expand_dims(np.max(evals,axis=1),1)).astype(int)

print(sklearn.metrics.confusion_matrix(np.argmax(evals_oh,axis=1), np.argmax(test_labels,axis=1)))
print('Weighted Accuracy', np.sum(evals_oh*test_labels)/4096)
         

