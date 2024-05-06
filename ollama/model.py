import ollama
from transformers import AutoTokenizer
import os

from split_doc import DocumentPreprocess

def run(task, model, tokenizer, max_length=2000):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    with open(f'documents/{task}.txt') as f:
            document = f.read()
    with open(f'prompts/{task}.txt') as f:
            prompt = f.read()

    system_prompt, user_prompt = prompt.split('*************')
            
    if task == 'legal':
        document_process = DocumentPreprocess()

        # split the documents into chunks
        legal_document, contract = document.split('----------------')
        legal_document_segments = document_process.main(legal_document, tokenizer, max_length)
        contract_segments = document_process.main(contract, tokenizer, max_length)
    
        results = []
        print(f'法規被拆成{len(legal_document_segments)}段, 合約被拆成{len(contract_segments)}段')
        
        
        for legal_seg in legal_document_segments:     
            contract_results = []
            for contract_seg in contract_segments:
                temp_prompt = prompt.replace('[LAW]', legal_seg)
                temp_prompt = temp_prompt.replace('[CONTRACT]', contract_seg)
                print(temp_prompt)
    
                # llama3:70b-instruct-q2_K
                response = ollama.chat(model=model, messages=[
                    {
                    'role': 'system',
                    'content': system_prompt
                    },
                    {
                    'role': 'user',
                    'content': temp_prompt
                    },
                ])
                contract_results.append(response['message']['content'])
                print(contract_results[-1])
                input()
                
            results.append('\n'.join(contract_results))
        # print(results[0])
        # print(results[1])
    
        # result = results[0]
        # for i in range(1, len(results)):
        #     print(results[i])
        #     input()
        #     merged_prompt = final_prompt + result + '\n------------\n' + results[i] + '\n請合併以上合約比對結果，並以清晰的格式呈現。[/INST]'
        #     result = generator(merged_prompt)[0]['generated_text'].split('[/INST]')[-1]
    
        # with open('results.txt', 'w') as f:
        #     f.write(result)
        return results
    elif task == 'violation':
        final_result = ''

        documents = document.split('----------------')
        print(len(documents))
        for i, doc in enumerate(documents):
            temp_prompt = user_prompt.replace('[text]', doc)
            response = ollama.chat(model=model, messages=[
                    {
                    'role': 'system',
                    'content': system_prompt
                    },
                    {
                    'role': 'user',
                    'content': temp_prompt
                    },
                ])
            if i != 0:
                final_result += '\n\n----------------\n\n'
        final_result += response['message']['content']
        return final_result

    elif task == 'translate':
        with open(f'documents/{task}.txt') as f:
            documents = f.readlines()
        translation = ''
        
        document = [documents[i:i+1] for i in range(0, len(documents), 1)]
        for i in range(len(document)):
            print(f'{i}/{len(document)}')
            doc = ''.join(document[i])
            prompt = user_prompt.replace('[text]', doc)
    
            # llama3:70b-instruct-q2_K
            response = ollama.chat(model=model, messages=[
                {
                'role': 'system',
                'content': system_prompt
                },
                {
                'role': 'user',
                'content': prompt
                },
            ])
            print(response['message']['content'])
            translation = translation + '\n' + response['message']['content']
        return translation

        
    else:
        user_prompt = user_prompt.replace('[text]', document)
    
        # llama3:70b-instruct-q2_K
        response = ollama.chat(model=model, messages=[
            {
            'role': 'system',
            'content': system_prompt
            },
            {
            'role': 'user',
            'content': user_prompt
            },
        ])

        
        return response['message']['content']


# python main.py violation
# python main.py keyword
# python main.py summary
# python main.py legal