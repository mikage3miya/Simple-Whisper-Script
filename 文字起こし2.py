
import sys
import os
from faster_whisper import WhisperModel



input_file = sys.argv[1]
audio_path = input_file
file_name, file_ext = os.path.splitext(input_file)

model_size = "large-v3"
model = WhisperModel(model_size, device="cpu",  compute_type="int8")

# run inference
segments, info = model.transcribe(audio_path, beam_size=5, without_timestamps=True)


outfilename = file_name + '.txt'
outfile = open(outfilename,"w",encoding='utf-8')

for segment in segments:
    outfile.write(f"[{segment.start:.2f} - {segment.end:.2f}]: {segment.text}") 
    outfile.write("\n") 
outfile.close()


