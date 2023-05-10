# Script for downstream classification of music genres using the "small" subset of FMA
# Usage: python3 downstream_genre.py path-to-dataset path-to-pretrained-model --train 
# if included, --train argument trains a shallow classifier upon the frozen encoder
# if not, it evaluates the respective shallow classifier
# eg. python3 downstream_genre.py /home/data/FMA models/vocals/  --train

import os
import argparse
import json
import collections #import Counter
import pandas as pd
import sys
import numpy as np
import tensorflow as tf
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import cola
from cola import constants
from mscol import network


def expand_labels(Y,n):
  Y = np.tile(Y,(n,1,1))
  Y = np.transpose(Y,(1,0,2))
  Yx = np.reshape(Y,(Y.shape[1]*Y.shape[0],Y.shape[2]))
  return Yx


def prepare_standard_example(x, is_training):
    if is_training:
        s = int(np.random.uniform(0,x.shape[0]-98))
        x = x[s:s+98,:]
    else:
        x = tf.signal.frame(x,frame_length=98,frame_step=98,axis=0,pad_end=False)
    return x

parser = argparse.ArgumentParser()

parser.add_argument('fma_path',type=str)
parser.add_argument('model_dir',type=str)
parser.add_argument('--train',default=False,action='store_true')
args = parser.parse_args()
fma_path = args.fma_path
model_dir = args.model_dir
train = args.train


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()


train_path = fma_path+'/melspecs/training'
valid_path = fma_path+'melspecs/validation'
test_path = fma_path+'/melspecs/test'

genre_dict = ['Hip-Hop','Pop','Instrumental','Rock','Electronic','International','Experimental','Folk']
csv_path = fma_path+'/fma_metadata/'
data_ = pd.read_csv(csv_path+'/tracks.csv')
data_small = data_.loc[data_.index[data_['set.1'] == 'small']]

train_dirs = os.listdir(train_path)
train_numel = len(train_dirs)-1
train_keys = []
cnt = 0
data_train = data_small.loc[data_small.index[data_small['set'] == 'training']]
ids = data_train["Unnamed: 0"]
genre_labels = data_train['track.7'].tolist()
idlist = ids.tolist()
idlist_str = [str(x).zfill(6) for x in idlist]
genres = np.zeros((train_numel,8))

for dir_ in train_dirs:
  if not dir_.startswith('g'): #?!
    idx = idlist_str.index(dir_.split('_')[0])
    gidx = genre_dict.index(genre_labels[idx])
    train_keys.append(dir_[:-4])
    genres[cnt,gidx] = 1
    cnt += 1

genres_train = genres[:cnt,:]

valid_dirs = os.listdir(valid_path)
valid_numel = len(valid_dirs)-1
valid_keys = []
cnt = 0
data_valid = data_small.loc[data_small.index[data_small['set'] == 'validation']]
ids = data_valid["Unnamed: 0"]
genre_labels = data_valid['track.7'].tolist()
idlist = ids.tolist()
idlist_str = [str(x).zfill(6) for x in idlist]
genres = np.zeros((valid_numel,8))

for dir_ in valid_dirs:
  if not dir_.startswith('g'):
    idx = idlist_str.index(dir_.split('_')[0])
    gidx = genre_dict.index(genre_labels[idx])
    valid_keys.append(dir_[:-4])
    genres[cnt,gidx] = 1
    cnt += 1
genres_valid = genres[:cnt,:]

test_dirs = os.listdir(test_path)
test_numel = len(test_dirs)-1
test_keys = []
cnt = 0
data_test = data_small.loc[data_small.index[data_small['set'] == 'test']]
ids = data_test["Unnamed: 0"]
genre_labels = data_test['track.7'].tolist()
idlist = ids.tolist()
idlist_str = [str(x).zfill(6) for x in idlist]
genres = np.zeros((test_numel,8))

for dir_ in test_dirs:
  if not dir_.startswith('g'):
    idx = idlist_str.index(dir_[:-4])
    gidx = genre_dict.index(genre_labels[idx])
    test_keys.append(dir_[:-4])
    genres[cnt,gidx] = 1
    cnt += 1
genres_test = genres[:cnt,:]

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
outputs = tf.keras.layers.Dense(8)(x) #shallow classifier for downstream tasks

model = tf.keras.Model(inputs, outputs)
model.get_layer("encoder").trainable = False #True for finetuning (False as default)
model.compile(
         optimizer=tf.keras.optimizers.Adam(0.0005),
          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.CategoricalAccuracy()])

checkpoint = tf.train.Checkpoint(model)

print(train)

if train:

  best_val_acc = 0
  patience = 0
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
      for n in range(0,batchSize):
        filename = train_path+'/'+train_keys[batch_comp[n]]+'.npy'

        y_raw = np.load(filename) #loads a 4-sec excerpt
        specs[n,:,:,:] = prepare_standard_example(y_raw,True)

      labels = genres_train[batch_comp[:batchSize],:]
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
      filename = valid_path+'/'+valid_keys[n]+'.npy'
      y_raw = np.load(filename)
      specs[n,:,:,:] = prepare_standard_example(y_raw,True)

    labels = genres_valid
    evals = model.evaluate(x=specs,y=labels,batch_size=128,verbose=0)
    print("Validation Loss:","{:.5f}".format(evals[0]),"Validation Accuracy","{:.5f}".format(evals[1]))
    if evals[1] > best_val_acc:
      best_val_acc = evals[1]
      patience = 0
      save_path = checkpoint.save('./'+model_dir[:-1]+'_FMA/'+'checkpoint')
    else:
      patience += 1
      if patience > 4:
        print('Epochs required for convergence of downstream model:',k-4)
        print('Performance of the downstream model in the training set', "{:.5f}".format(running_acc))
        print('Performance of the downstream model in the validation set', "{:.5f}".format(best_val_acc))
        break

model.load_weights(tf.train.latest_checkpoint(model_dir[:-1]+'_FMA/')).expect_partial()
specs = np.empty((30*test_numel,98,64,1))
for n in range(0,30*test_numel,30):
  filename = test_path+'/'+test_keys[n//30]+'.npy'
  y_raw = np.load(filename)
  specs[n:n+30,:,:,:] = prepare_standard_example(y_raw,False)

labels = expand_labels(genres_test,30)
evals = model.predict(specs)
evals = np.reshape(evals,(800,30,8))
evals = np.sum(evals,axis=1)
evals_oh = (evals == np.expand_dims(np.max(evals,axis=1),1)).astype(int)

print(np.sum(evals_oh), np.sum(genres_test))
print(sklearn.metrics.confusion_matrix(np.argmax(evals_oh,axis=1), np.argmax(genres_test,axis=1)))
print('weighted accuracy', np.sum(evals_oh*genres_test)/800)
         
print(np.sum(genres_test),np.sum(evals))

