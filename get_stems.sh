# Upsamples wav files in a given directory, and then extracts target sources
# Place it in the target directory!
# Usage: ./get_stems.sh source_data_dir

for d in $1'/'*; do
  p=${d%????}
  echo $p'_up.wav'
  sox $d -r 44100 $p'_up.wav'
  umx $p'_up.wav' --targets 'vocals' 'drums' 'bass' 'other'
done
