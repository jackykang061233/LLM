from download_video import download_video
from faster_whisper_run import faster_whisper_single_file
from large_model import normal_model

import argparse
import os

import time

from dotenv import load_dotenv
from huggingface_hub import login


# login to hf hub
load_dotenv()
api_token = os.environ.get('HUGGINGFACE_API_TOKEN')
login(api_token)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('id',
                        help='youtube影片id',
                        type=str)
    parser.add_argument('model_id',
                        help='模型id',
                        type=str)
    parser.add_argument('output',
                        help='輸出檔案名稱',
                        type=str)
    args = parser.parse_args()

    start = time.time()
    # download_video(args.id) 
    faster_whisper_single_file(audio=f'downloaded_videos/{args.id}.wav', output='audio_output', prompt='')

    with open(f'audio_output/{args.id}_1.txt') as f:
        file = f.read()  

#     prompt = f'Please translate the following excerpt from the Federal Open Market Committee (FOMC) press conference transcript into Traditional Chinese. Given the length of the transcript, proceed with the translation in sections, each not exceeding 500 words. Ensure the translation is accurate, maintaining the original meaning and the correct use of technical terms.
 
# [text]
 
# Once the translation of the provided segment is complete, please supply the Traditional Chinese version of the paragraph.'
    generation_args = {
        "max_new_tokens": 8000,
        "return_full_text": True,
        "temperature": 0.0,
        "do_sample": False,
    }
    # result = normal_model(args.model_id, args.model_id, prompt, generation_args)
    # with open(args.output, 'w') as f:
    #     f.write(result)
    # print(result)

    
    

    prompt = f"You are a summary assistant. You provide a concise and comprehensive summary of the given text. The summary should capture the main points and key details of the text while conveying the author' intended meaning accurately. If the data is mentioned, please provide data or number to back up the summary stance. Please ensure that the summary is well-organized and easy to read, with clear headings and subheadings to guide the reader through each section. The length of the summary should be appropriate to capture the main points and key details of the text, without including unnecessary information or becoming overly long. Text:\n {file}"

    # mistralai/Mistral-7B-Instruct-v0.2
    # microsoft/Phi-3-mini-128k-instruct
    result = normal_model(args.model_id, args.model_id, prompt, generation_args) 
    with open(f'{args.output}.txt', 'w') as f:
        f.write('FOMC 5/1\n\n')
        f.write(result)

    prompt = f"Please analyze the attached transcript from the following FOMC press conference. Focus on identifying and summarizing key discussions and decisions related to the following topics:\nInflation: Mention any references to 'inflation', 'inflation expectations', 'inflation target', 'CPI' (Consumer Price Index), 'PCE' (Personal Consumption Expenditures), and 'price stability'.\nLabor Market: Look for discussions on the 'labor market', 'unemployment rate', 'hiring', 'quit rates', 'maximum employment', and 'wage growth'.\nMonetary Policy: Highlight any mentions of the 'policy rate', 'neutral rate'.\nFinancial market: 'financial conditions'.\nEconomic Outlook: Note any comments on 'confidence', their 'data-dependent' approach, 'progress' towards goals, 'productivity', and how these relate to the FOMC's 'mandate'.\nYour summary should be concise, prioritizing the extraction of significant points related to these keywords. Please provide bullet points for clarity and quote directly from the transcript. Also return the main theme of the transcipt\nTranscipt:\n {file}"
    
    result = normal_model(args.model_id, args.model_id, prompt, generation_args)
    with open('fomc_1.txt', 'a') as f:
        f.write('\n\n---------------\n\nSummary by keywords:\n\n')
        f.write(result)


    end = time.time()
    print(f"transcribe cost time(s):{end - start} seconds" )
    
    # prompt = f"Please read the given text carefully. After reading, identify and list the most relevant keywords that capture the essence of the topics discussed. The keywords should be single words or short phrases that represent significant concepts within the text. Provide the keywords in a list format using bullet points. Text:\n {file}"
    # messages = [
    #     {"role": "user", "content": prompt},
    # ]
    
    # result = normal_model("microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-128k-instruct", messages)
    # with open('keyword.txt', 'w') as f:
    #     f.write(result)
    
    # prompt = f"Please analyze the following FOMC press conference transcript and return the line numbers for the first and last lines of each journalist's section, including their question and the Fed Chairman's response. Journalists typically introduce themselves with phrases like like 'Uh,' 'Hi,' or the journalist's name followed by their affiliation, and end before the next journalist starts speaking.\nTranscipt:\n {file}"
    # messages = [
    #     {"role": "user", "content": prompt},
    # ]
    
    # result = normal_model("microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-128k-instruct", messages)
    # with open('identify.txt', 'w') as f:
    #     f.write(result)
    
    