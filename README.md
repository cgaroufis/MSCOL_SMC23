# Multi-Source Contrastive Learning from Musical Audio

This repository contains the code for reproducing the results and experiments of the paper "Multi-Source Contrastive Learning from Musical Audio", submitted to the 2023 Sound and Music Computing Conference (SMC-2023). We also provide pre-trained models for either finetuning, or direct training of shallow classifiers for downstream tasks.

Links for datasets:
- MTAT: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
- NSynth: https://magenta.tensorflow.org/datasets/nsynth
- FMA (*small* subset): https://github.com/mdeff/fma 

The code used borrows heavily from the COLA repository: https://github.com/google-research/google-research/tree/master/cola

The default train/validation/test splits were used for NSynth and FMA; for MTAT, we followed the data cleaning and split used in https://github.com/jongpillee/music_dataset_split/tree/master/MTAT_split

Open-Unmix was used for acquiring the various source excerpts: https://github.com/sigsep/open-unmix-pytorch

Python dependencies for the preprocessing of the source/downstream datasets and the training/evaluation of the contrastive framework and the downstream classifiers (alternatively, you can set up a complete conda environment using the uploaded mscol.yml file):

librosa (0.8.1)  
numpy (1.21.6)  
pandas (1.4.4)  
scipy (1.9.1)  
scikit-learn (1.1.1)  
tensorflow-gpu (2.4.1)

## Data Preprocessing

To get mel-spectrograms from raw audio files, you can simply use the preprocess_{dataset}.py files. The preprocess_multitracks.py computes the mel-spectorgrams from the available source excerpts.

Since open-unmix was used for acquiring the various source excerpts, it is advised to install the open-unmix package via pip (ffmpeg is also required). Open-unmix operates at a sampling frequency of 44.1 kHz, accepting as input .wav files. To make your dataset compatible with openunmix:

- (optionally) run get_wavs.sh to transform the dataset into .wav files
- run get_stems.sh to resample each wav file, and acquire stems corresponding to the bass, the drums, the vocals and the melodic accompaniment of each song excerpt
- run downsample_wavs.sh to downsample each wav file to 16000 kHz, to enable further preprocessing.

## Model Pretraining

To pre-train an encoder, run the pretraining.py script:

python3 pretraining.py datapath model dir ... 

## Shallow Classifier Training

To train a shallow classifier on top of a pre-trained encoder at a specific downstream task, ...
