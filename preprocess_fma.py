import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from cola import data
import librosa

# pre-computes mel-spectrograms and saves them in npy format, to improve loading speed.
# accepts as input the directory to be mel'd.
# usage: python3 preprocess_fma.py path-to-FMA

datapath = sys.argv[1]

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()

genre_dict = ['Hip-Hop','Pop','Instrumental','Rock','Electronic','International','Experimental','Folk']
base_path = datapath+'/fma_small/' 
csv_path = datapath+'/fma_metadata/'
target_dir = datapath+'/melspecs/'

data_ = pd.read_csv(csv_path+'tracks.csv')
data_small = data_.loc[data_.index[data_['set.1'] == 'small']]

for subset in ['training','validation','test']:
  label_distr = np.zeros((8,))
  ct = 0
  data_sub = data_small.loc[data_small.index[data_small['set'] == subset]]
  ids = data_sub["Unnamed: 0"]
  genre_onehot = np.zeros((len(ids),8))
  genre_labels = data_sub['track.7'].tolist()
  idlist = ids.tolist()
   
  os.makedirs(target_dir+subset,exist_ok=True)
  for k in range(0,len(ids)):
    idx = genre_dict.index(genre_labels[k])
    label_distr[idx] += 1
    genre_onehot[k,idx] = 1
    filename = (str(idlist[k]).zfill(6))
    foldername = filename[:3]
    full_name = base_path+foldername+'/'+filename+'.mp3'
    print(full_name)
    if filename not in ['099134','108925','133297']: #empty/corrupt files
      y,fs=librosa.load(full_name,sr=44100)
      z = librosa.resample(y,orig_sr=44100,target_sr=16000)
      z = tf.math.l2_normalize(z, epsilon=1e-9)
      y = data.extract_log_mel_spectrogram(z)
      if subset != 'test':
        for i in range(0,y.shape[0]-392,392):
          ys = y[i:i+392,:]
          ys = ys[Ellipsis, tf.newaxis]
          filename_ = target_dir+subset+'/'+filename+'_'+str(i//392)+'.npy'
          np.save(filename_,ys)
          ct += 1
      else:
        y = y[Ellipsis,tf.newaxis]
        filename_ = target_dir+subset+'/'+filename+'.npy'
        np.save(filename_,y)
        ct += 1
    
      print(ct, 'melspectrograms saved')
  
