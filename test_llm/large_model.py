import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from ctransformers import AutoModelForCausalLM as cAutoModelForCausalLM
import time


# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def quantization_model(model_id, model_file, model_type, prompt):
    llm = cAutoModelForCausalLM.from_pretrained(model_id, model_file=model_file, model_type=model_type, gpu_layers=50, context_length=30000)
    print(llm(prompt))

def normal_model(model_id, tokenizer_id, prompt, generation_args):
    torch.random.manual_seed(0)
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    
    output = pipe(messages, **generation_args)
    return output[0]['generated_text'][1]['content']


# with open('doc/fed.txt') as f:
#     file = f.read()
    
# prompt = f"You are a summary assistant. You provide a concise and comprehensive summary of the given text. The summary should capture the main points and key details of the text while conveying the author' intended meaning accurately. Please ensure that the summary is well-organized and easy to read, with clear headings and subheadings to guide the reader through each section. The length of the summary should be appropriate to capture the main points and key details of the text, without including unnecessary information or becoming overly long. Text:\n {file}"
# messages = [
#     {"role": "user", "content": prompt},
# ]

# result = normal_model("microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-128k-instruct", messages)
# with open('fomc_press.txt', 'a') as f:
#     f.write('FOMC 5/1\n\n')
#     f.write(result)

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

# prompt = f"Please analyze the attached transcript from the following FOMC press conference. Focus on identifying and summarizing key discussions and decisions related to the following topics:\nInflation: Mention any references to 'inflation', 'inflation expectations', 'inflation target', 'CPI' (Consumer Price Index), 'PCE' (Personal Consumption Expenditures), and 'price stability'.\nLabor Market: Look for discussions on the 'labor market', 'unemployment rate', 'hiring', 'quit rates', 'maximum employment', and 'wage growth'.\nMonetary Policy: Highlight any mentions of the 'interest rate', 'policy rate', 'neutral rate'.\nFinancial market: 'financial conditions'.\nEconomic Outlook: Note any comments on 'confidence', their 'data-dependent' approach, 'progress' towards goals, 'productivity', and how these relate to the FOMC's 'mandate'.\nYour summary should be concise, prioritizing the extraction of significant points related to these keywords. Please provide bullet points for clarity and quote directly from the transcript. Also return the main theme of the transcipt\nTranscipt:\n {file}"
# messages = [
#     {"role": "user", "content": prompt},
# ]

# result = normal_model("microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-128k-instruct", messages)
# with open('fomc_press.txt', 'a') as f:
#     f.write('\n---------------\nSummary by keywords:\n')
#     f.write(result)




#quantization_model("TheBloke/Nous-Capybara-34B-GGUF", "nous-capybara-34b.Q4_K_M.gguf", "yi", prompt)

# from dotenv import load_dotenv
# from huggingface_hub import login


# # login to hf hub
# load_dotenv()
# api_token = os.environ.get('HUGGINGFACE_API_TOKEN')
# login(api_token)
