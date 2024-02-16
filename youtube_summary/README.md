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
## usage
### 格式
```
python3 main.py youtube-url openai-key prompt
```
### 例子
```
python3 main.py https://www.youtube.com/watch?v=ImrKxlLJCEY openai-key multi-speakers
```
總共有三個arguments
* arg1(youtube-url): 要下載的youtube網址
* arg2(api_key): openai的api-key
* arg3(prompt): 有兩種(預設為multi-speakers)
  1. multi-speakers
  2. single-speaker

  **兩者的差別就在最後連接GPT所給的指令不同**

## summary
下載的summary會儲存在summary資料夾，檔名為youtube影片id(網址中在watch?v=後的)
