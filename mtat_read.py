import numpy as np
import pandas as pd

incsv = pd.read_csv("annotations_final.csv",delimiter='\t')
print(incsv.head())

top50 = ["guitar", "classical", "slow", "techno", "strings", "drums", "electronic", "rock", "fast", "piano", "ambient", "beat", "violin", "vocal", "synth", "female", "indian", "opera", "male", "singing", "vocals", "no vocals", "harpsichord", "loud", "quiet", "flute", "woman", "male vocal", "no vocal", "pop", "soft", "sitar", "solo", "man", "classic", "choir", "voice", "new age", "dance", "male voice", "female vocal", "beats", "harp", "cello", "no voice", "weird", "country", "metal", "female voice", "choral"]
incsv_filt = incsv[top50]
incsv_filt_paths = incsv["mp3_path"]

incsv_filt_valid = incsv_filt[incsv["mp3_path"].str.startswith('c')]
incsv_filt_test = incsv_filt[incsv["mp3_path"] > 'd']
incsv_filt_train = incsv_filt[incsv["mp3_path"] < 'c']

incsv_filt_paths_valid = incsv_filt_paths[incsv_filt_paths.str.startswith('c')]
incsv_filt_paths_test = incsv_filt_paths[incsv["mp3_path"] > 'd']
incsv_filt_paths_train = incsv_filt_paths[incsv["mp3_path"] < 'c']

np.save('y_valid.npy',incsv_filt_valid)
np.save('y_train.npy',incsv_filt_train)
np.save('y_test.npy',incsv_filt_test)

np.save('x_valid.npy',incsv_filt_paths_valid,allow_pickle=True)
np.save('x_train.npy',incsv_filt_paths_train,allow_pickle=True)
np.save('x_test.npy',incsv_filt_paths_test,allow_pickle=True)

print(incsv_filt_valid.shape, incsv_filt_test.shape, incsv_filt_train.shape)

