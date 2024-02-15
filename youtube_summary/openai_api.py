from openai import OpenAI
client = OpenAI(api_key="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")

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

def call_openai_api(id, prompt_example):
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
    )

    with open(f"summary/{id}.txt", "w") as f:
        f.write(response.choices[0].message.content)
    print("Complete!")
