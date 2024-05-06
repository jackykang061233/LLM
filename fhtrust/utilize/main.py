import os
from dotenv import load_dotenv
from huggingface_hub import login

from config.core import config
from model import load_pipeline, violation_investigation, extract_keyword, summarization, legal_compliance

import time
import argparse


# login to hf hub
load_dotenv()
api_token = os.environ.get('HUGGINGFACE_API_TOKEN')
login(api_token)

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    start_time = time.time()
    # list of task functions
    task_functions = {
        'summary': summarization,
        'keyword': extract_keyword,
        'violation': violation_investigation,
        'legal': legal_compliance,
    }

    # parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('task',
                        help='任務',
                        type=str)
    args = parser.parse_args()

    
    with open(f'document/{args.task}.txt') as f:
        document = f.read()
    
    if config.model.quantized:
        model = config.model.quantized_model.name
        generator = load_pipeline(model_name=model,
                                  tokenizer_name=config.model.quantized_model.tokenizer,
                                  quantized=True,
                                  quantized_model_file=config.model.quantized_model.modelfile, 
                                  quantized_model_type=config.model.quantized_model.modeltype)
    else:
        model = config.model.non_quantized_model.name
        generator = load_pipeline(model_name=model)

    with open(f'prompts/{args.task}.txt') as f:
        prompt = f.read()

    if args.task in task_functions:
        result = task_functions[args.task](prompt=prompt, document=document, generator=generator, 
                                           model=model, max_length=2000, final_prompt=config.final_prompt)
    else:
        result = "No such task"

    print(result)
    print(time.time() - start_time)

# python main.py violation
# python main.py keyword
# python main.py summary
# python main.py legal


