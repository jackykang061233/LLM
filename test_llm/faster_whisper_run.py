from faster_whisper import WhisperModel
import torch
 
import time
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ['REQUESTS_CA_BUNDLE'] = r"D:\重要\fuhwatrust-FHDC1-CA.crt"
 
def faster_whisper_single_file(audio, output, prompt="以下是繁體中文的為復華投信的測試，包含集保", model='large-v3'):
    # check if you have a GPU available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    print("DEVICE:"+DEVICE)
 
    #load Whipser model
    print(f"start to load model" )
    start = time.time()
    model = WhisperModel(model, device=DEVICE, compute_type=COMPUTE_TYPE)
    end = time.time()
    print(f"load model cost time(s):{end - start} seconds" )
 
    print(f"start to  transcribe" )
    start = time.time()
    transcription = ''
    segments, _ = model.transcribe(audio, initial_prompt=prompt)
    segments = [seg.text for seg in segments]
    output = output + "/" + audio.split("/")[-1].split('.')[0] + ".txt"

    process_text(segments, output)
    end = time.time()
    print(f"transcribe cost time(s):{end - start} seconds" )
 
def process_text(lines, output_file):
    with open(output_file, 'w') as file:
        buffer = ''
        for line in lines:
            if line.strip() and not line.strip().endswith('.'):
                buffer += line.strip() + ' '
            else:
                buffer += line.strip()
                file.write(buffer + '\n')
                buffer = ''
                
# if __name__ == "__main__":
#     from download_video import download_video
#     id = 'ohuJH9fld3w'
#     download_video(id) 

#     faster_whisper_single_file(audio=f'downloaded_videos/{id}.wav', output='audio_output', prompt='')