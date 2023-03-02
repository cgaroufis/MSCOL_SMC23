import os
import sys
import numpy as np

import tensorflow as tf
from cola import data
import librosa

# pre-computes mel-spectrograms and saves them in npy format, to improve loading speed.
# also computes spectrograms for the 4 sources as computed by open-unmix
# accepts as input the directory to be mel'd.
# usage: python3 preprocess_multitracks.py path-to-source-wavs path-to-vanilla-MTT train/valid/test

base_path = sys.argv[1] #directory that contains the extracted source files
full_path = sys.argv[2] #directory that contains the dataset
set_ = sys.argv[3]
file_ = open('./data-split/'+set_+'_list_pub.cP') #contains the valid npy files (the .pub files in ./data_split)
dest_path = full_path+'/melspecs_src/'
os.makedirs(dest_path+set_,exist_ok=True)

ct = 0

for line in file_:
  if 'npy' in line: 
    idx = line.find("'")
    key = line[idx+1:-6]
    path_to_folder = base_path+'/'+key+'_up_open-unmix'
    if os.path.isdir(path_to_folder):
      for ikey in ['bass','drums','other','vocals']:
        x,fs = librosa.load(path_to_folder+'/'+ikey+'_down.wav',sr=16000)
        z = tf.math.l2_normalize(x, epsilon=1e-9) #we keep the prenormalized to a separate variable to not generate silent stems.
        y = data.extract_log_mel_spectrogram(z)
        cn = 0
        if set_ != 'test': #in practice ran only for training/validation sets
          for i in range(0,y.shape[0]-392,392):
            if np.mean(np.abs(x[cn:cn+64000])) > 0.01:
              ys = y[i:i+392,:]
              ys = ys[Ellipsis, tf.newaxis]
              filename_ = dest_path+set_+'/'+key[2:]+'_'+str(i//392)+'_'+ikey+'.npy'
              np.save(filename_,ys)
              ct += 1
            cn += 64000

    print(ct, 'full source melspecs saved')
