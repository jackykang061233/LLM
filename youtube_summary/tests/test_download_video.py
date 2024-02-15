import pytest 
from download_video import download_video
from pytube.exceptions import RegexMatchError, VideoUnavailable
import os

def test_valid_url():
    url = "https://www.youtube.com/watch?v=MU4ubyVgpsc"
    download_video(url, "test")
    assert os.path.isfile("downloaded_videos/test.wav")
    
def test_invalid_url():
    with pytest.raises((RegexMatchError, VideoUnavailable)):
        url = "https://www.youtube.com/watch?v=MU4ubyVgps"
        download_video(url, "test")