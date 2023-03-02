# Multi-Source Contrastive Learning from Musical Audio

This repository contains the code for reproducing the results and experiments of the paper "Multi-Source Contrastive Learning from Musical Audio", submitted to the 2023 Sound and Music Computing Conference (SMC-2023). We also provide pre-trained models for either finetuning, or direct training of shallow classifiers for downstream tasks.

Links for datasets:
- MTAT: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
- NSynth: https://magenta.tensorflow.org/datasets/nsynth
- FMA (*small* subset): https://github.com/mdeff/fma 

The code used borrows heavily from the COLA repository: https://github.com/google-research/google-research/tree/master/cola

The default train/validation/test splits were used for NSynth and FMA; for MTAT, we followed the data cleaning and split used in https://github.com/jongpillee/music_dataset_split/tree/master/MTAT_split

Open-Unmix was used for acquiring the various source excerpts: https://github.com/sigsep/open-unmix-pytorch

Dependencies for the preprocessing of the source/downstream datasets and the training/evaluation of the contrastive framework and the downstream classifiers (alternatively, you can set up a complete conda environment using the uploaded mscol.yml file):

librosa (0.8.1)  
numpy (1.21.6)  
pandas (1.4.4)  
scipy (1.9.1)  
scikit-learn (1.1.1)  
tensorflow-gpu (2.4.1)

Important! The open-unmix model operates at a sampling frequency of 44.1 kHz and accepts .wav files. Bash scripts are provided for the upsampling of the audio excerpts and the downsampling of the separated sources before further preprocessing.

