from download_video import download_video
from openai_api import call_openai_api
import subprocess
import argparse
import re

def run_diarization_command(id):
    command = ["python", "whisper-diarization/diarize_parallel.py", "-a", f"downloaded_videos/{id}.mp3"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)

def get_youtube_id(youtube_url):
    regex = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)'
    match = re.match(regex, youtube_url)
    
    return match.group(1)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('youtube_url', type=str, help='A YouTube URL to process.')
    parser.add_argument('api_key', type=str, help='The openai api key.')
    parser.add_argument('prompt', type=str, nargs='?', default='multi-speakers', help='Prompt for summarization.')
    args = parser.parse_args()

    id = get_youtube_id(args.youtube_url)
    
    download_video(id)
    run_diarization_command(id)
    call_openai_api(id, args.prompt, args.api_key)


    # url = "https://www.youtube.com/watch?v=ImrKxlLJCEY"
    # url = "https://www.youtube.com/watch?v=gRTBRV2_S5Q"
