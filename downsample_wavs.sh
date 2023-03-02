#Script for downsampling the extracted source .wav files to 16kHz
#Usage: ./downsample_wavs.sh data_dir

for d in $1'/'*; do
  for f in $d'/'*; do
    p=${f::-4}
    if [[ $f != *'down.wav' ]]; then
      echo $f
      sox $f -r 16000 $p'_down.wav'
      rm $f
    fi
  done
done
