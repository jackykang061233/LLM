from pyannote.audio import Pipeline


pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="")

import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("downloaded_videos/test.mp3")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
