#Transform all .mp3 files in a given subdirectory into .wav files
#Usage: ./get_wavs.sh data_dir
#Dependencies: ffmpeg

for d in $1'/'*; do
  echo $d 
  p=${d::-4}
  ffmpeg -i $d $p'.wav'
done
