from pytube import YouTube
from pytube.exceptions import RegexMatchError, VideoUnavailable
from pydub import AudioSegment

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context 

url = "https://www.youtube.com/watch?v=gRTBRV2_S5Q"
url = "https://www.youtube.com/watch?v=QLK4_wYNTe"

def download_video(url, video_name):
    def onProgress(stream, chunk, remains):
        total = stream.filesize                     
        percent = (total-remains) / total * 100     
        print(f"Downloading… {percent:05.2f}", end="\r")
        
    try:
        yt = YouTube(url, on_progress_callback=onProgress)
        print("download...")
    
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(output_path="downloaded_videos/", filename=f"{video_name}.mp3")
        
        try:
            # Load the audio file
            audio = AudioSegment.from_file(f"downloaded_videos/{video_name}.mp3")
            print("Audio file loaded successfully.")
        except Exception as e:
            print("Error loading audio file:", e)
    
    except (RegexMatchError, VideoUnavailable) as e:
        print("Error:", e)
        print("Invalid YouTube URL or video unavailable.")
        raise e




