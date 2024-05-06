from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ctransformers import AutoModelForCausalLM as cAutoModelForCausalLM
import torch

from keybert.llm import TextGeneration
from keybert import KeyLLM

from typing import List
from split_doc import DocumentPreprocess

def load_pipeline(model_name: str, 
                  tokenizer_name: str=None,
                  quantized: bool=False,
                  task: str='text-generation', 
                  max_new_tokens: int=6000, 
                  repetition_penalty: float=1.1, 
                  top_p: float=0.00,
                  quantized_model_file: str=None, 
                  quantized_model_type: str=None, 
                  context_length: int=8000,
                  gpu_layers: int=50) -> pipeline:
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.uint8
    tokenizer = model_name if not tokenizer_name else tokenizer_name

    if quantized:
        print('You are using quantized model')
        print(quantized_model_file, quantized_model_type)
        model = cAutoModelForCausalLM.from_pretrained(
            model_name,
            model_file=quantized_model_file,
            model_type=quantized_model_type,
            context_length=context_length,
            gpu_layers=gpu_layers)
    else: 
        print('You are using unquantized model')
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    # Pipeline
    generator = pipeline(
        model=model, tokenizer=tokenizer,
        task=task,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    return generator


def violation_investigation(prompt: str, document: str, generator: pipeline, **kwargs) -> List:    
    final_result = ''

    documents = document.split('----------------')
    for i, doc in enumerate(documents):
        temp_prompt = prompt.replace('[DOCUMENT]', doc)
        result = generator(temp_prompt)[0]['generated_text']
        result = result.split('[/INST]')[-1]
        if i != 0:
            final_result += '\n\n----------------\n\n'
        final_result += result

    return final_result

def extract_keyword(prompt: str, document: str, generator: pipeline, **kwargs) -> List:    
    llm = TextGeneration(generator, prompt=prompt)
    
    kw_model = KeyLLM(llm)
    keywords = kw_model.extract_keywords(document)

    return keywords[0][0]
    # print(keywords[0][0])
    # lines = [keywords[i][0].split('\n') for i in range(len(keywords))]
    # keywords_list = [[l.split('.', 1)[1] for l in line] for line in lines]
    
    # return keywords_list

def summarization(prompt: str, document: str, generator: pipeline, **kwargs) -> List:    
    prompt = prompt.replace('[DOCUMENT]', document)
    
    summary = generator(prompt)[0]['generated_text']
    summary = summary.split('[/INST]')[-1]

    return summary

def legal_compliance(prompt: str, document: str, generator: pipeline, **kwargs) -> List:
    document_process = DocumentPreprocess()
    model = kwargs.get('model')
    max_length = kwargs.get('max_length')
    final_prompt = kwargs.get('final_prompt')

    legal_document, contract = document.split('----------------')
    legal_document_segments = document_process.main(legal_document, model, max_length)
    contract_segments = document_process.main(contract, model, max_length)

    results = []
    print(f'法規被拆成{len(legal_document_segments)}段, 合約被拆成{len(contract_segments)}段')
    for legal_seg in legal_document_segments:     
        contract_results = []
        for contract_seg in contract_segments:
            temp_prompt = prompt.replace('[DOCUMENT]', legal_seg)
            temp_prompt = temp_prompt.replace('[CONTRACT]', contract_seg)

            input()
            result = generator(temp_prompt)[0]['generated_text']
            result = result.split('[/INST]\n')[-1]
            result = result.replace('\n\n', '\n')
            contract_results.append(result)
        results.append('\n'.join(contract_results))

    result = results[0]
    for i in range(1, len(results)):
        print(results[i])
        input()
        merged_prompt = final_prompt + result + '\n------------\n' + results[i] + '\n請合併以上合約比對結果，並以清晰的格式呈現。[/INST]'
        result = generator(merged_prompt)[0]['generated_text'].split('[/INST]')[-1]

    with open('results.txt', 'w') as f:
        f.write(result)

    return result