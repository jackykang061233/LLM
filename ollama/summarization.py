# transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.notebook import tqdm
import torch

# llama_index
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document

# config
from dotenv import load_dotenv
from huggingface_hub import login
import os

# login to hf hub
load_dotenv()
api_token = os.environ.get('HUGGINGFACE_API_TOKEN')
login(api_token)
os.environ["CUDA_VISIBLE_DEVICES"]="0"


model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.bfloat16)
    
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map=0,
                                             quantization_config=quantization_config)
model.eval()


def recursive_summarization(document, tokenizer, model, generate_kwargs, chunk_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def recursive_summarization_helper(nodes, device):
        summarized_texts = []
        for node in tqdm(nodes):
            prompt = f"""###System:
You are an expert agent in informatino extraction and summarization.
### User:
Read the following context document:
-------------
{node.get_content()}
-------------
Your tasks are as follows:
1.-Write an extensive, fluid, and continuous paragraph summarizing the most important aspects of the information you have read.
2.-You can only synthesize your response using exclusively the information from the context document.
### Assistant:
According to the context information, the summary is: """
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            tokens = model.generate(**inputs, **generate_kwargs)

            completion_tokens = tokens[0][inputs['input_ids'].size(1):]
            response_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            summarized_texts.append(response_text)

        if len(nodes) == 1:
            return summarized_texts[0]
        else:
            for t in summarized_texts:
                with open('summarized_texts.txt', 'a') as f:
                    f.write('\n'+t+'\n')
            return final_summary(summarized_texts, tokenizer, model, {})

    def final_summary(docs, tokenizer, model, generate_kwargs):
        print('final summary')
        prompt = f""" The following is set of summaries, take these and distill it into a final, consolidated summary:\n"""
        for i, doc in enumerate(docs):
            prompt = prompt + f'Document {i+1}' + doc + '\n'
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        tokens = model.generate(**inputs, **generate_kwargs)

        completion_tokens = tokens[0][inputs['input_ids'].size(1):]
        response_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return response_text
        

    node_parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    document = Document(text=document)
    initial_nodes = node_parser.get_nodes_from_documents([document])
    return recursive_summarization_helper(initial_nodes, device)

    
with open('documents/UYnc6bsgkJQ.txt') as f:
    document = f.read()


result = recursive_summarization(document=document, tokenizer=tokenizer, model=model, generate_kwargs={}, chunk_size=512)
with open('result.txt', 'w') as f:
    f.write(result)


