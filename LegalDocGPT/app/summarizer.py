import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def summarize_contract(clause_data):
    prompt = f"Summarize the following contract analysis in plain language:\n\n{clause_data}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    summary = response['choices'][0]['message']['content']
    return summary
