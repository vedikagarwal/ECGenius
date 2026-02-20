import csv
import json
import time
import re
from openai import OpenAI, OpenAIError

API_KEY = "AIzaSyD_TaTYsYLspqyRR-O_iKtFj0XmYpy5tXM"
llama_key = "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"

def getRephrased2(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=llama_key,
    )

    while True:
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "openrouter.ai",
                    "X-Title": "openrouter.ai",
                },
                model="meta-llama/llama-3-8b-instruct:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return completion.choices[0].message.content

        except OpenAIError as e:
            if "rate limit" in str(e).lower():
                print("Rate limit exceeded. Retrying in 1 minute and 10 seconds...")
                time.sleep(70)
            else:
                raise e

def remove_whitespace_and_newlines(text):
    return text.replace(" ", "").replace("\n", "")

def getSummary(text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=" + API_KEY
    prompt_summary = (
        "Summarize the following row of medical data. Focus on key aspects like patient details, group, heart rate, ultrasound system, video details, and RV metrics. Provide the summary in one concise sentence between ---.""### {} ###"
    )

    prompt = prompt_summary.format(text)
    response = getRephrased2(prompt)
    print(response)
    return response

def process_data(input_details):
        
    with open('summarized_data.json', 'w') as data:
        data.write("[")
        row_text = json.dumps(input_details, ensure_ascii=False)
        summary = getSummary(row_text).replace('---', '').strip()
                
        fileWrite = {
                'input': row_text,
                'summary': summary
                }
                
        data.write(
            re.sub(r"Here is the rephrased text.*?(:|\\n\\n)", "", 
            str(fileWrite).replace("'", "\"")) + ', \n'
            )
            
        data.write("]")

    return summary

def main():
    input_details = input("Enter the patient details: ")
    process_data(input_details)
    print(f"Processing complete. Results saved to 'summarized_data.json'")

if __name__ == "__main__":
    main()
