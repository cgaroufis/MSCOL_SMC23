# Multi-Source Contrastive Learning from Musical Audio

This repository contains the code for reproducing the results and experiments of the paper "Multi-Source Contrastive Learning from Musical Audio", submitted to the 2023 Sound and Music Computing Conference (SMC-2023).

Links for datasets:
- MTAT: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
- NSynth: https://magenta.tensorflow.org/datasets/nsynth
- FMA (*small* subset): https://github.com/mdeff/fma 

The code used borrows heavily from the COLA repository: https://github.com/google-research/google-research/tree/master/cola

The default train/validation/test splits were used for NSynth and FMA; for MTAT, we followed the data cleaning and split used in https://github.com/jongpillee/music_dataset_split/tree/master/MTAT_split

Open-Unmix was used for acquiring the various source excerpts: https://github.com/sigsep/open-unmix-pytorch

Dependencies:

numpy

tensorflow (2.4.0)

umx-gpu

To reproduce the experiments with degraded/randomly assigned sources, the dependencies of open-unmix are required. 
