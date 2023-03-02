import os
import sys
import numpy as np

import tensorflow as tf
from cola import data
import librosa

# pre-computes mel-spectrograms and saves them in npy format, to improve loading speed.
# accepts as input the directory to be mel'd.
# Usage: python3 preprocess_nsynth.py path-to-nsynth nsynth-train/nsynth-valid/nsynth-test

base_path = sys.argv[1]
set_ = sys.argv[2] #training, validation, testing
full_path = base_path+set_+'.jsonwav/'+set_+'/audio'
os.makedirs(base_path+set_+'.jsonwav/'+set_+'/melspecs',exist_ok = True)

wavs = os.listdir(full_path)
ct = 0
for note in wavs:
  x,sr = librosa.load(full_path+'/'+note,sr=16000)
  x = tf.math.l2_normalize(x, epsilon=1e-9)
  y = data.extract_log_mel_spectrogram(x)
  y = y[Ellipsis, tf.newaxis]
  filename_ = base_path+set_+'.jsonwav/'+set_+'/melspecs/'+note[:-4]+'.npy' 
  np.save(filename_,y)
  ct += 1
  print(ct,"mel spectrograms saved")
