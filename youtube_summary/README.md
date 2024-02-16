# 將youtube影片轉換成summary

## installation
必須要先下載Cython和FFMPEG

### Cython
```
pip install cython
```
或
```
sudo apt update && sudo apt install cython3
```
### FFMPEG
根據不同作業系統不一樣
```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```
最後下載必要packages
```
pip install -r requirements.txt
```
