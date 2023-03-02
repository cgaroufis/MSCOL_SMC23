import os
import sys
import numpy as np

import tensorflow as tf
from cola import data
import librosa

# pre-computes mel-spectrograms for MTAT and saves them in npy format, to improve loading speed.
# accepts as inputs a) the directory of MTAT, b) the subset (train-validation-test)
# saves the spectrograms in a ./melspecs/{train,validation,test} subdirectory, trimmed in 4-sec excerpts

# eg. python3 preprocess.py path-to-MTAT train

base_path = sys.argv[1]  #path to the directory MagnaTagATune is stored.
set_ = sys.argv[2] 
file_ = open('./data-split/'+set_+'_list_pub.cP') #filters the valid entries
os.makedirs(base_path+'/melspecs/'+set_,exist_ok=True) #target dir

print(file_)

ct = 0

for line in file_:
  if len(line) > 0:
    idx = line.find("'") #contains the "valid" entries from MTT
    key = line[idx+1:-6]
    path_to_file = base_path+'/'+key+'.wav' 
    if os.path.isfile(path_to_file):
      z,fs = librosa.load(path_to_file,sr=16000)
      z = tf.math.l2_normalize(z, epsilon=1e-9)
      y = data.extract_log_mel_spectrogram(z)
      if set_ != 'test':
        for i in range(0,y.shape[0]-392,392): #392 frames approximately equals 4 seconds.
          ys = y[i:i+392,:]
          ys = ys[Ellipsis, tf.newaxis]
          filename_ = base_path+'melspecs/'+set_+'/'+key[2:-4]+'_'+str(i//392)+'_full.npy'
          np.save(filename_,ys)
          ct += 1
      else:
        y = y[Ellipsis,tf.newaxis]
        filename_ = base_path+'melspecs/'+set_+'/'+key[2:-4]+'_full.npy'
        np.save(filename_,y)
        ct += 1

    print(ct, 'full melspectrograms saved')
