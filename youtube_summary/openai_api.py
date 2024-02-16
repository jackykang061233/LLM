from openai import OpenAI
from dotenv import load_dotenv

def load_text(id):
    with open(f"downloaded_videos/{id}.txt") as f: 
        text = f.read()
    return text

def prompt(prompt_example):
    if prompt_example == "multi-speakers":
        with open("prompt/multiple_speakers.txt") as f: 
            text = f.read()
    else:
        text = "Write a concise summary of the following text:"
    
    return text

def call_openai_api(id, prompt_example, api_key):
    client = OpenAI(api_key=api_key)
    print("Connecting to openai...")
    response = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      messages=[
        {
          "role": "system",
          "content": prompt(prompt_example)
        },
        {
          "role": "user",
          "content": load_text(id)
        }
      ],
      temperature=0.0,
      top_p=0.0001,
    )

    with open(f"summary/{id}.txt", "w") as f:
        f.write(response.choices[0].message.content)
    print(f"The summary is saved in the folder summary with the name {id}.txt")
    print("Complete!")
