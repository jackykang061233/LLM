import ollama
import argparse

import time
import os
from dotenv import load_dotenv
from huggingface_hub import login

from model import run

# login to hf hub
load_dotenv()
api_token = os.environ.get('HUGGINGFACE_API_TOKEN')
login(api_token)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    start_time = time.time()
    
    # parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('task',
                        help='任務',
                        type=str)
    parser.add_argument('--max_length',
                        help='文件分割長度',
                        default=2000,
                        type=int,
                        required=False)
    parser.add_argument('--model',
                        help='模型名稱',
                        default='llama3',
                        type=str,
                        required=False)
    parser.add_argument('--tokenizer',
                        help='模型名稱',
                        default='meta-llama/Meta-Llama-3-8B',
                        type=str,
                        required=False)
    args = parser.parse_args()

    result = run(args.task, args.model, args.tokenizer, args.max_length)
    # with open('translation_oneline.txt', 'w') as f:
    #     f.write(result)
    print(result)
    print(time.time() - start_time)

# python main.py violation
# python main.py keyword
# python main.py summary
# python main.py legal