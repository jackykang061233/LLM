from pytube import YouTube
from pytube.exceptions import RegexMatchError, VideoUnavailable
from pydub import AudioSegment

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context 

def download_video(id):
    def onProgress(stream, chunk, remains):
        total = stream.filesize                     
        percent = (total-remains) / total * 100     
        print(f"Downloading… {percent:05.2f}", end="\r")
        
    try:
        url = "https://www.youtube.com/watch?v=" + id
        yt = YouTube(url, on_progress_callback=onProgress)
        print("download...")
    
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(output_path="downloaded_videos/", filename=id+".mp3")
        
    except (RegexMatchError, VideoUnavailable) as e:
        print("Error:", e)
        print("Invalid YouTube URL or video unavailable.")
        raise e





